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

A shared ledger (``tests/test_helpers/fuzz_ledger.py``) manages tracked
wrong-answer and crash-class findings. It is currently empty because gh #3547
and #3957 are fixed and covered by regressions.

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

Routing coverage (three modes, axes ``routing_method`` x ``routing_input_mode`` x ``logits_dtype``):
  * **pre-routed** (RoutingInputMode.PackedPrecomputed): the host computes the top-k per method and
    feeds packed indices -- the original path.
  * **unpacked pre-routed** (RoutingInputMode.UnpackedPrecomputed): TRTLLM FP4 receives separate
    int32 ids + BF16 or FP32 weights without packed-id construction.
  * **in-kernel** (RoutingInputMode.FromLogits): the kernel routes from raw logits per
    RoutingConfig.method -- reaches the bug cluster the pre-routed harness structurally can't:
    DeepSeekV3 group-topk + bias (#2575), all-negative logits (#2822), fp32 router logits (#2796),
    bias-method weight leakage (#2485/#2907). The SAME ``_route`` oracle (ported verbatim from the
    kernel-validated references in ``tests/moe/test_trtllm_gen_fused_moe.py``) is the authority for
    every mode, so a kernel that routes wrong is caught by check #2. In-kernel routing is
    single-GPU (non-EP) here; EP + in-kernel routing semantics are a separate validation.

Coverage today: NVFP4, BF16, block/per-tensor FP8, MXFP4/W4A16, and MxInt4.
CuteDSL NVFP4 is pre-routed-only; FromLogits and UnpackedPrecomputed restrict
dispatch to capable TRTLLM runners. MxInt4 covers packed and BF16-FromLogits
routing.

ENABLED BY DEFAULT: this suite runs like any other test. Unsupported configurations skip at the
no-wired-backend check. FLASHINFER_UMOE_FUZZ=0 remains the emergency waiver.
Run it explicitly:
  CUDA_HOME=<cuda> CUDA_VISIBLE_DEVICES=<sm100-idx> pytest tests/moe/test_unified_moe_fuzz.py
NOTE: `pytest --forked` does NOT work here (CUDA inits at collection ->
"Cannot re-initialize CUDA in forked subprocess"); for crash-isolated enumeration run each
test id in its own process instead (see var/03-ssh-docker-workflow.md).
Env: FLASHINFER_UMOE_FUZZ_NUM_TESTS (default 160), FLASHINFER_UMOE_FUZZ_SEED (default 0),
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
  * New backend -> add its config class to the matching dtype handler's ``candidate_configs``;
    the live ``_BACKEND_RUNNERS`` registry then supplies the runner and architecture gate. If it
    ships with a tracked bug, add one ledger ``Finding`` (a non-quarantine case still RUNS; an
    xpass then hard-fails until the entry is removed).
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
  * Bugs THIS fuzzer found + filed:
    - fixed gh #3547 (trtllm EP offset>0 all-zero).
    - fixed gh #3957 (cumulative corruption).
    - open gh #3548 (activation global-scale gap == roadmap #5's scale-policy fix).
  * Findings writeups: flashinfer_triage/EP_OFFSET_FINDING.md, flashinfer_triage/WEIGHT_SCALE_FINDING.md.
  * The unified API under test: PR #3093 (branch ``moe_api``); this fuzzer is PR aleozlx/flashinfer#6
    (branch ``yanxu/unified-moe-api-fuzzer``).
"""

from __future__ import annotations

import functools
import hashlib
import os
import random
from dataclasses import dataclass
from typing import Callable

import pytest
import torch
import torch.nn.functional as F

from flashinfer.autotuner import autotune
from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.fp4_quantization import fp4_quantize
from flashinfer.fused_moe import (
    MoEActivationPack,
    MoELayer,
    MoEWeightPack,
    RoutingInputMode,
)
from flashinfer.fused_moe.api import (
    # Typed activation values
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    ReLU,
    ReLU2,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
    # Unified configs and backend options
    BackendOptions,
    B12xNvfp4Config,
    B12xW4A16Config,
    CutlassBf16Config,
    CutlassFp8BlockConfig,
    CutlassFp8PerTensorConfig,
    CutlassHummingConfig,
    CutlassMxfp8Config,
    CutlassMxfp8Mxfp4Config,
    CutlassNvfp4Config,
    CutlassW4A8Config,
    CutlassW4A16Config,
    CuteDslConfig,
    ExecutionConfig,
    ExpertConfig,
    MoEConfig,
    MoEFinalizeConfig,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
    TrtllmBf16Config,
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
    TrtllmFp8PerTensorConfig,
    TrtllmMxInt4Config,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.fused_moe.runners import _TrtllmRunnerBase
from flashinfer.fused_moe.prepare import _quantize_mxfp4_linear
from flashinfer.jit.cpp_ext import get_cuda_version
from flashinfer.quantization import e2m1_and_ufp8sf_scale_to_float
from flashinfer.quantization.fp8_quantization import mxfp8_quantize
from flashinfer.tllm_enums import RoutingMethodType
from flashinfer.utils import get_compute_capability

from tests.test_helpers.fuzz_ledger import FuzzLedger

NUM_TESTS = int(os.environ.get("FLASHINFER_UMOE_FUZZ_NUM_TESTS", "160"))
# Debug knob: comma-separated backend_key allowlist (e.g. "cute_dsl") to run a
# backend-scoped sequence -- used to bisect cross-call state corruption by backend (gh #3957).
_BACKEND_FILTER = {
    b for b in os.environ.get("FLASHINFER_UMOE_FUZZ_BACKENDS", "").split(",") if b
}
# Debug knob: skip the autotune(True) production-path step entirely -- used to isolate whether
# cross-call corruption accumulates in the profiling path (cudagraph captures) or the plain
# forward/tactic path (gh #3957).
_NO_AUTOTUNE = os.environ.get("FLASHINFER_UMOE_FUZZ_NO_AUTOTUNE", "0") not in ("", "0")
BASE_SEED = int(os.environ.get("FLASHINFER_UMOE_FUZZ_SEED", "0"))
# Perfect-repro hook: if set (comma-separated seeds), the suite runs ONLY those configs. A curated
# seed maps to its hand-written Cfg; any other seed is regenerated via the deterministic _gen(seed),
# so a single seed reproduces exactly one config. The repro command printed on every test uses this.
_ONLY_SEEDS = os.environ.get("FLASHINFER_UMOE_FUZZ_ONLY_SEED", "")

# --- CI gate: ON by default (FLASHINFER_UMOE_FUZZ=0 is the emergency waiver) ----------------
# History: this suite was opt-in (FLASHINFER_UMOE_FUZZ=1) while (a) gh #3547 was open and (b) the
# accumulated single-process run could hit `CUDA error: device-side assert triggered` ->
# `Fatal Python error: Aborted` (2026-06-09 triage). Both are now understood (2026-07-14, full
# default run on a B200-class SM100): #3547 is fixed (its EP-offset configs pass), and the abort
# is root-caused mechanically -- an async device-side assert from one config poisons the CUDA
# context and the pending c10 error escapes a destructor at interpreter shutdown
# (std::terminate). It is not a separate Heisenbug: any assert-class *finding* ends this fuzzer's
# pytest process after the failure is reported. The shard_group marker keeps the accumulated
# sequence together in one pytest invocation, preserving the regression while the sharding runner
# can still isolate failures in other groups. The historical gh #3957 finding
# (a silent OOB write with a moving victim) was fixed by gh #4186; keeping this accumulated
# sequence enabled is its regression coverage.
# Set FLASHINFER_UMOE_FUZZ=0 to disable in an emergency; FLASHINFER_UMOE_FUZZ=1 (the old opt-in
# value) still enables and is now a no-op.
pytestmark = pytest.mark.skipif(
    os.environ.get("FLASHINFER_UMOE_FUZZ", "1") == "0",
    reason="unified MoE fuzzer disabled via FLASHINFER_UMOE_FUZZ=0",
)

# Per-backend determinism contract, established empirically (CRC across reruns) + confirmed against
# code. A "True" backend MUST reproduce bitwise; flip to False only with evidence (and ideally an
# upstream note), because a deterministic->non-deterministic regression is exactly a bug to catch.
_DETERMINISTIC = {
    "cutlass_bf16": True,
    "cutlass_w4a16": True,
    "trtllm_fp4_routed": True,  # bitwise-stable across reruns in calibration
    "cute_dsl": False,  # atomic scatter-add finalize -> non-bit-exact by design
    "trtllm_bf16_routed": True,  # same trtllm-gen finalize path as fp4_routed; bitwise-stable in calibration
    "trtllm_fp8_block": True,
    "trtllm_fp8_per_tensor": True,
}
# The seven #4610 CUTLASS runners and both b12x runners are intentionally
# absent until repeated-run bitwise calibration is completed on SM90 and
# SM120/SM121 respectively.

# Known-bug ledger (shared mechanism: tests/test_helpers/fuzz_ledger.py). Two severities:
# quarantine=False entries are RUN with a tolerated wrong answer (xpass flags the fix);
# quarantine=True entries are xfailed up front and never launch (crash / device-state class --
# one such config poisons the CUDA context for every later test in the process).
LEDGER = FuzzLedger(
    "unified-moe",
    findings=(
        # Finding(match=..., reason="...; gh #NNNN", quarantine=..., backend=...)
        # Wrong-answer entries: empty since the gh #3547 EP-offset double-subtraction fix.
        # The historical gh #3957 cross-call corruption had a moving victim and could not be
        # quarantined by config predicate. It was fixed by gh #4186, so no ledger entry remains;
        # this accumulated sequence is the regression test.
    ),
)


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
    make_act_pack: (
        Callable | None
    )  # (x, selected_experts, final_scales) -> pre-routed pack
    make_act_pack_logits: (
        Callable | None  # (x, routing_logits, routing_bias) -> pack (in-kernel routing)
    )
    reference: Callable  # (x, w1, w2, selected_experts, final_scales, I) -> fp32 [T,H] authority
    poison: Callable  # in-place fill a kernel-owned output buffer with garbage + (NaN/Inf if repr.)
    out_dtype: torch.dtype  # output buffer dtype (used to locate it in the inputs list)
    atol_frac: float  # numeric tolerance vs reference = atol_frac * ‖ref‖∞
    rtol: float
    # Contract overlays use backend-native activation/preparation recipes and
    # therefore need a reference derived after the prepared view exists.
    post_prepare_reference: Callable | None = None
    prepare_weights: Callable | None = None
    weight_snap: Callable | None = None


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
    # Raw bf16 activations. Use BF16-grid routing weights represented as FP32
    # so TRTLLM's packed-id path and CUTLASS's separate-weight path share one
    # exact semantic input.
    return MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=None,
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
        topk_ids=selected_experts,
        topk_weights=final_scales.to(torch.bfloat16).float(),
    )


def _bf16_act_pack_logits(x, routing_logits, routing_bias):
    return MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=None,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=routing_logits,
        routing_bias=routing_bias,
    )


def _mxint4_act_pack_logits(x, routing_logits, routing_bias):
    return MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=None,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=routing_logits,
        routing_bias=routing_bias,
    )


def _apply_typed_activation(fc1, activation, intermediate_size):
    """Apply one typed activation to canonical GEMM1 output."""
    if isinstance(activation, ReLU2):
        return F.relu(fc1) ** 2
    if isinstance(activation, Identity):
        return fc1
    if isinstance(activation, GELU):
        return F.gelu(fc1, approximate="none")
    if isinstance(activation, ReLU):
        return F.relu(fc1)
    if isinstance(activation, SiLU):
        return F.silu(fc1)

    up, gate = fc1[:, :intermediate_size], fc1[:, intermediate_size:]
    if isinstance(activation, SwiGLU):
        gate = gate.clamp(max=activation.limit)
        up = up.clamp(min=-activation.limit, max=activation.limit)
        return gate * torch.sigmoid(activation.alpha * gate) * (up + activation.beta)
    if isinstance(activation, GeGLU):
        return F.gelu(gate, approximate="none") * up
    if isinstance(activation, GeGLUTanh):
        return F.gelu(gate, approximate="tanh") * up
    if isinstance(activation, SwiGLUStep):
        return F.silu(gate).clamp(max=activation.limit) * up.clamp(
            min=-activation.limit, max=activation.limit
        )
    if isinstance(activation, SiTU):
        if activation.clamp_limit is not None:
            up = up.clamp(min=-activation.clamp_limit, max=activation.clamp_limit)
            gate = gate.clamp(max=activation.clamp_limit)
        linear = (
            up
            if activation.linear_scale is None
            else activation.linear_scale * torch.tanh(up / activation.linear_scale)
        )
        return (
            linear
            * activation.gate_scale
            * torch.tanh(gate / activation.gate_scale)
            * torch.sigmoid(gate)
        )
    raise AssertionError(f"unsupported fuzz activation {activation!r}")


def _bf16_reference(
    x,
    w1,
    w2,
    selected_experts,
    final_scales,
    intermediate_size,
    expert_offset=0,
    activation=None,
):
    """Dense bf16 MoE authority: same SwiGLU convention as ``_nvfp4_reference``
    but no fp4 requant -- the only intermediate quantization is the bf16 rounding
    of the gemm1 and gemm2 outputs, mirrored below.  Routing weights are cast through bf16
    to match the packed-id truncation in pack_inputs, so the tolerance measures
    kernel error, not oracle mismatch."""
    final_scales = final_scales.to(torch.bfloat16).float()
    activation = activation or SwiGLU()
    x32, half = x.float(), intermediate_size
    out = torch.zeros_like(x32)
    for local_e in range(w1.shape[0]):
        mask = selected_experts == local_e + expert_offset
        if not mask.any():
            continue
        tok, nth = torch.where(mask)
        fc1 = x32[tok] @ w1[local_e].float().t()
        inter = _apply_typed_activation(fc1, activation, half)
        inter = inter.to(torch.bfloat16).float()  # gemm1 output is stored bf16
        expert_out = (inter @ w2[local_e].float().t()).to(torch.bfloat16).float()
        out[tok] += final_scales[tok, nth, None] * expert_out
    return out


def _block_fp8_dequant(x_q, scale, variant):
    if variant is QuantVariant.DeepSeekFp8:
        if x_q.dim() == 2:
            expanded = scale.transpose(0, 1).repeat_interleave(128, dim=-1)
        else:
            expanded = scale.repeat_interleave(128, dim=-2).repeat_interleave(
                128, dim=-1
            )
        return x_q.float() * expanded
    scale_f32 = torch.pow(2.0, scale.to(torch.uint8).float() - 127.0)
    return x_q.float() * scale_f32.repeat_interleave(32, dim=-1)


def _mxfp8_quant_matrix(x):
    """Quantize a logical matrix without applying the MoE weight shuffle."""
    q, scale = mxfp8_quantize(x, is_sf_swizzled_layout=False)
    return q, scale.view(torch.uint8).reshape(x.shape[0], x.shape[1] // 32)


def _mxfp4_quant_dequant_matrix(x):
    """Torch MXFP4 round-trip that is valid on both Hopper and Blackwell."""
    q, sf = _quantize_mxfp4_linear(x.to(torch.bfloat16).contiguous())
    low = q & 0xF
    high = q >> 4
    codes = torch.stack((low, high), dim=-1).reshape(x.shape).to(torch.long)
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=x.device,
        dtype=torch.float32,
    )
    values = magnitudes[codes & 0x7]
    values = torch.where((codes & 0x8) != 0, -values, values)
    scales = torch.exp2(sf.to(torch.int16).to(torch.float32) - 127)
    return values * scales.repeat_interleave(32, dim=-1)


def _mxfp4_snap(t: torch.Tensor, *, bf16_activation: bool) -> torch.Tensor:
    if t.dim() == 2:
        if bf16_activation:
            return t.to(torch.bfloat16)
        q, sf = _mxfp8_quant_matrix(t.to(torch.bfloat16))
        return _block_fp8_dequant(q, sf, QuantVariant.MxFp8).to(torch.bfloat16)
    return torch.stack([_mxfp4_quant_dequant_matrix(expert) for expert in t]).to(
        torch.bfloat16
    )


def _mxfp4_act_pack(x, selected_experts, final_scales, *, variant: QuantVariant):
    q, sf = TrtllmFp4Config.prepare_activations(x, variant=variant)
    return MoEActivationPack(
        hidden_states_q=q,
        hidden_states_scale=sf,
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
        topk_ids=selected_experts,
        # TRTLLM's packed-id ABI rounds routing weights through BF16, whereas
        # CUTLASS consumes FP32. Supplying BF16-grid values in FP32 gives both
        # backends one exact routing-weight contract.
        topk_weights=final_scales.to(torch.bfloat16).float(),
    )


def _mxfp4_act_pack_logits(x, routing_logits, routing_bias, *, variant):
    q, sf = TrtllmFp4Config.prepare_activations(x, variant=variant)
    return MoEActivationPack(
        hidden_states_q=q,
        hidden_states_scale=sf,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=routing_logits,
        routing_bias=routing_bias,
    )


def _mxfp4_reference(
    x,
    w1,
    w2,
    selected_experts,
    final_scales,
    intermediate_size,
    expert_offset=0,
    *,
    variant,
):
    if variant is QuantVariant.MXFP4:
        x_q, x_sf = _mxfp8_quant_matrix(x)
        x32 = _block_fp8_dequant(x_q, x_sf, QuantVariant.MxFp8)
    else:
        x32 = x.float()
    w1_32 = torch.stack([_mxfp4_quant_dequant_matrix(expert) for expert in w1])
    w2_32 = torch.stack([_mxfp4_quant_dequant_matrix(expert) for expert in w2])
    final_scales = final_scales.to(torch.bfloat16).float()
    out = torch.zeros_like(x32)
    for local_e in range(w1.shape[0]):
        token, slot = torch.where(selected_experts == local_e + expert_offset)
        if token.numel() == 0:
            continue
        up = x32[token] @ w1_32[local_e, :intermediate_size].t()
        gate = x32[token] @ w1_32[local_e, intermediate_size:].t()
        inter = F.silu(gate) * up
        if variant is QuantVariant.MXFP4:
            inter_q, inter_sf = _mxfp8_quant_matrix(inter.to(torch.bfloat16))
            inter = _block_fp8_dequant(inter_q, inter_sf, QuantVariant.MxFp8)
        else:
            inter = inter.to(torch.bfloat16).float()
        out[token] += final_scales[token, slot, None] * (inter @ w2_32[local_e].t())
    return out


def _block_fp8_act_pack(x, selected_experts, final_scales, *, variant):
    q, sf = TrtllmFp8BlockConfig.prepare_activations(x, variant=variant)
    return MoEActivationPack(
        hidden_states_q=q,
        hidden_states_scale=sf,
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
        topk_ids=selected_experts,
        topk_weights=final_scales,
    )


def _block_fp8_snap(t: torch.Tensor) -> torch.Tensor:
    """Keep FP8 fuzz inputs in a realistic MoE numerical range."""
    scale = 0.02 if t.dim() == 3 else 0.25
    return (t * scale).to(torch.bfloat16)


def _block_fp8_act_pack_logits(x, routing_logits, routing_bias, *, variant):
    q, sf = TrtllmFp8BlockConfig.prepare_activations(x, variant=variant)
    return MoEActivationPack(
        hidden_states_q=q,
        hidden_states_scale=sf,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=routing_logits,
        routing_bias=routing_bias,
    )


def _block_fp8_reference(
    x,
    w1,
    w2,
    selected_experts,
    final_scales,
    intermediate_size,
    expert_offset=0,
    *,
    variant,
):
    if variant is QuantVariant.DeepSeekFp8:
        x_q, x_sf = TrtllmFp8BlockConfig.prepare_activations(x, variant=variant)
    else:
        x_q, x_sf = _mxfp8_quant_matrix(x)
    x32 = _block_fp8_dequant(x_q, x_sf, variant)
    if variant is QuantVariant.DeepSeekFp8:
        view = TrtllmFp8BlockConfig.prepare_weights(
            w1,
            w2,
            variant=variant,
            num_local_experts=w1.shape[0],
            hidden_size=x.shape[1],
            intermediate_size=intermediate_size,
            device=x.device,
        )
        w1_32 = _block_fp8_dequant(
            view["gemm1_weights"], view["gemm1_weights_scale"], variant
        )
        w2_32 = _block_fp8_dequant(
            view["gemm2_weights"], view["gemm2_weights_scale"], variant
        )
    else:
        w1_32 = torch.stack(
            [
                _block_fp8_dequant(q, sf, variant)
                for q, sf in (_mxfp8_quant_matrix(expert) for expert in w1)
            ]
        )
        w2_32 = torch.stack(
            [
                _block_fp8_dequant(q, sf, variant)
                for q, sf in (_mxfp8_quant_matrix(expert) for expert in w2)
            ]
        )
    final_scales = final_scales.to(torch.bfloat16).float()
    out = torch.zeros_like(x32)
    for local_e in range(w1.shape[0]):
        token, slot = torch.where(selected_experts == local_e + expert_offset)
        if token.numel() == 0:
            continue
        up = x32[token] @ w1_32[local_e, :intermediate_size].t()
        gate = x32[token] @ w1_32[local_e, intermediate_size:].t()
        inter = F.silu(gate) * up
        if variant is QuantVariant.DeepSeekFp8:
            inter_q, inter_sf = TrtllmFp8BlockConfig.prepare_activations(
                inter.to(torch.bfloat16), variant=variant
            )
        else:
            inter_q, inter_sf = _mxfp8_quant_matrix(inter.to(torch.bfloat16))
        inter = _block_fp8_dequant(inter_q, inter_sf, variant)
        expert_out = inter @ w2_32[local_e].t()
        out[token] += final_scales[token, slot, None] * expert_out
    return out


def _fp8_per_tensor_global_scale(x):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    amax = x.float().abs().amax()
    return torch.where(amax > 0, fp8_max / amax, torch.ones_like(amax))


def _fp8_per_tensor_act_pack_logits(x, routing_logits, routing_bias):
    input_scale = _fp8_per_tensor_global_scale(x)
    q, sf = TrtllmFp8PerTensorConfig.prepare_activations(
        x, hidden_states_scale_global=input_scale
    )
    return MoEActivationPack(
        hidden_states_q=q,
        hidden_states_scale=sf,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=routing_logits,
        routing_bias=routing_bias,
    )


def _fp8_per_tensor_act_pack(x, selected_experts, final_scales):
    input_scale = _fp8_per_tensor_global_scale(x)
    q, sf = TrtllmFp8PerTensorConfig.prepare_activations(
        x, hidden_states_scale_global=input_scale
    )
    assert sf is None
    return MoEActivationPack(
        hidden_states_q=q,
        hidden_states_scale=None,
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
        topk_ids=selected_experts,
        topk_weights=final_scales,
    )


def _fp8_per_tensor_dequant_experts(weights):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    amax = weights.float().abs().amax(dim=(-1, -2))
    scales = torch.where(amax > 0, fp8_max / amax, torch.ones_like(amax))
    q = (weights.float() * scales[:, None, None]).clamp(-fp8_max, fp8_max)
    return q.to(torch.float8_e4m3fn).float() / scales[:, None, None]


def _fp8_per_tensor_reference(
    x,
    w1,
    w2,
    selected_experts,
    final_scales,
    intermediate_size,
    expert_offset=0,
):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    final_scales = final_scales.to(torch.bfloat16).float()
    input_scale = _fp8_per_tensor_global_scale(x)
    intermediate_scale = torch.tensor(64.0, device=x.device)
    x_q = (x.float() * input_scale).clamp(-fp8_max, fp8_max)
    x32 = x_q.to(torch.float8_e4m3fn).float() / input_scale
    w1_32 = _fp8_per_tensor_dequant_experts(w1)
    w2_32 = _fp8_per_tensor_dequant_experts(w2)

    out = torch.zeros_like(x32)
    for local_e in range(w1.shape[0]):
        token, slot = torch.where(selected_experts == local_e + expert_offset)
        if token.numel() == 0:
            continue
        up = x32[token] @ w1_32[local_e, :intermediate_size].t()
        gate = x32[token] @ w1_32[local_e, intermediate_size:].t()
        inter = F.silu(gate) * up
        inter_q = (inter * intermediate_scale).clamp(-fp8_max, fp8_max)
        inter = inter_q.to(torch.float8_e4m3fn).float() / intermediate_scale
        expert_out = (inter @ w2_32[local_e].t()).to(torch.bfloat16).float()
        out[token] += final_scales[token, slot, None] * expert_out
    return out


def _mxint4_quant_dequant(weights):
    blocks = weights.float().reshape(-1, 32)
    block_max = blocks.amax(dim=-1, keepdim=True) * (8.0 / 7.0)
    block_min = blocks.amin(dim=-1, keepdim=True)
    scales = torch.maximum(block_max, -block_min) / 8.0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = (blocks / scales).round().clamp(-8, 7)
    stored_scales = scales.to(torch.bfloat16).float()
    return (quantized * stored_scales).reshape_as(weights)


def _mxint4_reference(
    x,
    w1,
    w2,
    selected_experts,
    final_scales,
    intermediate_size,
    expert_offset=0,
):
    x32 = x.float()
    w1_32 = _mxint4_quant_dequant(w1)
    w2_32 = _mxint4_quant_dequant(w2)
    final_scales = final_scales.to(torch.bfloat16).float()
    out = torch.zeros_like(x32)
    for local_e in range(w1.shape[0]):
        token, slot = torch.where(selected_experts == local_e + expert_offset)
        if token.numel() == 0:
            continue
        fc1 = x32[token] @ w1_32[local_e].t()
        inter = F.silu(fc1[:, intermediate_size:]) * fc1[:, :intermediate_size]
        inter = inter.to(torch.bfloat16).float()
        expert_out = (inter @ w2_32[local_e].t()).to(torch.bfloat16).float()
        out[token] += final_scales[token, slot, None] * expert_out
    return out


def _contract_bf16_act_pack(x, selected_experts, final_scales):
    """Packed CUTLASS/b12x contract: BF16 x and FP32 routing weights."""
    return MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=None,
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
        topk_ids=selected_experts,
        topk_weights=final_scales.float(),
    )


def _contract_fp8_act_pack(config_cls):
    def make(x, selected_experts, final_scales):
        q, scale = config_cls.prepare_activations(x)
        return MoEActivationPack(
            hidden_states_q=q,
            hidden_states_scale=scale,
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
            topk_ids=selected_experts,
            topk_weights=final_scales.float(),
        )

    return make


def _semantic_reference(
    x, w1, w2, selected_experts, final_scales, intermediate_size, activation
):
    """Semantic CUTLASS/b12x reference in canonical [up, gate] row order.

    Keep routing weights in FP32: these runners consume separate precomputed
    tensors, unlike TRTLLM's packed-ID path, which truncates weights to BF16.
    Quant-specific callers supply dequantized activation and weight operands.
    This idealized semantic oracle also leaves GEMM intermediates in FP32;
    ``_bf16_reference`` adds its backend-specific BF16 storage round-trips.
    """
    x32 = x.float()
    out = torch.zeros_like(x32)
    for expert in range(w1.shape[0]):
        token, slot = torch.where(selected_experts == expert)
        if token.numel() == 0:
            continue
        fc1 = x32[token] @ w1[expert].float().t()
        inter = _apply_typed_activation(fc1, activation, intermediate_size)
        expert_out = inter @ w2[expert].float().t()
        out[token] += final_scales[token, slot, None].float() * expert_out
    return out


def _dequant_cutlass_nvfp4(packed, scale):
    rows, packed_cols = packed.shape
    return e2m1_and_ufp8sf_scale_to_float(
        packed.cpu(),
        scale.cpu().reshape(-1),
        torch.ones(1, dtype=torch.float32),
        16,
        1,
        True,
    ).view(rows, packed_cols * 2)


def _dequant_cutlass_nvfp4_experts(packed, scales, device):
    return torch.stack(
        [_dequant_cutlass_nvfp4(packed[i], scales[i]) for i in range(packed.shape[0])]
    ).to(device=device, dtype=torch.bfloat16)


def _dequant_int4_grouped(packed, scale, group_size=128):
    even = packed.to(torch.int16) & 0xF
    odd = packed.to(torch.int16) >> 4
    even = torch.where(even >= 8, even - 16, even)
    odd = torch.where(odd >= 8, odd - 16, odd)
    unpacked = torch.stack((even, odd), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    return unpacked.float() * scale.float().repeat_interleave(group_size, dim=-1)


def _dequant_linear_mxfp4(packed, scales):
    low, high = packed & 0xF, packed >> 4
    codes = torch.stack((low, high), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=packed.device,
        dtype=torch.float32,
    )
    values = magnitudes[codes.long() & 0x7]
    values = torch.where((codes & 0x8) != 0, -values, values)
    scale = torch.exp2(scales.to(torch.int16).float() - 127)
    return values * scale.repeat_interleave(32, dim=-1)


def _mxfp8_quant_dequant_experts(weight):
    from flashinfer import mxfp8_dequantize_host
    from flashinfer.quantization.fp8_quantization import mxfp8_quantize

    dequantized = []
    for expert in range(weight.shape[0]):
        packed, scale = mxfp8_quantize(
            weight[expert], is_sf_swizzled_layout=True, alignment=32
        )
        dequantized.append(
            mxfp8_dequantize_host(
                packed.cpu().view(torch.uint8),
                scale.cpu().view(torch.uint8).reshape(-1),
                True,
            )
        )
    return torch.stack(dequantized).to(weight.device)


def _cutlass_post_reference(backend_key):
    def reference(
        x, w1, w2, selected_experts, final_scales, intermediate_size, view, activation
    ):
        x_ref, w1_ref, w2_ref = x, w1, w2
        if backend_key == "cutlass_nvfp4":
            one = torch.ones(1, device=x.device)
            x_q, x_sf = fp4_quantize(
                x, global_scale=one, sf_vec_size=16, is_sf_swizzled_layout=True
            )
            x_ref = _dequant_cutlass_nvfp4(x_q, x_sf).to(x.device)
            w1_ref = _dequant_cutlass_nvfp4_experts(
                view["fc1_expert_weights"], view["fc1_weight_block_scale"], x.device
            )
            w2_ref = _dequant_cutlass_nvfp4_experts(
                view["fc2_expert_weights"], view["fc2_weight_block_scale"], x.device
            )
        elif backend_key == "cutlass_fp8_per_tensor":
            x_ref = view["_activation_q"].float() * view["_activation_scale"]
            w1_ref = (
                view["fc1_expert_weights"].float() * view["fc1_dequant"][:, None, None]
            )
            w2_ref = (
                view["fc2_expert_weights"].float() * view["fc2_dequant"][:, None, None]
            )
        elif backend_key == "cutlass_fp8_block":
            w1_ref = view["fc1_expert_weights"].float() * view[
                "fc1_block_scale"
            ].repeat_interleave(128, -2).repeat_interleave(128, -1)
            w2_ref = view["fc2_expert_weights"].float() * view[
                "fc2_block_scale"
            ].repeat_interleave(128, -2).repeat_interleave(128, -1)
        elif backend_key == "cutlass_mxfp8_mxfp4":
            from flashinfer import mxfp4_dequantize, mxfp8_dequantize_host

            x_ref = mxfp8_dequantize_host(
                view["_activation_q"].cpu().view(torch.uint8),
                view["_activation_scale"].cpu().view(torch.uint8).reshape(-1),
                True,
            ).to(x.device)
            w1_ref = torch.stack(
                [
                    mxfp4_dequantize(
                        view["fc1_expert_weights"][i].cpu(),
                        view["fc1_expert_scales"][i]
                        .cpu()
                        .view(torch.uint8)
                        .reshape(-1),
                    )
                    for i in range(w1.shape[0])
                ]
            ).to(x.device)
            w2_ref = torch.stack(
                [
                    mxfp4_dequantize(
                        view["fc2_expert_weights"][i].cpu(),
                        view["fc2_expert_scales"][i]
                        .cpu()
                        .view(torch.uint8)
                        .reshape(-1),
                    )
                    for i in range(w2.shape[0])
                ]
            ).to(x.device)
        elif backend_key == "cutlass_mxfp8":
            from flashinfer import mxfp8_dequantize_host

            x_ref = mxfp8_dequantize_host(
                view["_activation_q"].cpu().view(torch.uint8),
                view["_activation_scale"].cpu().view(torch.uint8).reshape(-1),
                True,
            ).to(x.device)
            # Re-quantize independently from the canonical weights instead of
            # consuming the prepared scale view. This catches a broken
            # _pack_mxfp8_weight_scales transformation in the production path.
            w1_ref = _mxfp8_quant_dequant_experts(w1)
            w2_ref = _mxfp8_quant_dequant_experts(w2)
        elif backend_key == "cutlass_w4a8":
            from flashinfer.fused_moe.prepare import _quantize_int4_grouped

            q1, s1 = _quantize_int4_grouped(w1)
            q2, s2 = _quantize_int4_grouped(w2)
            w1_ref, w2_ref = (
                _dequant_int4_grouped(q1, s1),
                _dequant_int4_grouped(q2, s2),
            )
        elif backend_key == "cutlass_humming":
            q1, s1 = _quantize_mxfp4_linear(w1.reshape(-1, w1.shape[-1]))
            q2, s2 = _quantize_mxfp4_linear(w2.reshape(-1, w2.shape[-1]))
            w1_ref = _dequant_linear_mxfp4(q1, s1).view_as(w1)
            w2_ref = _dequant_linear_mxfp4(q2, s2).view_as(w2)
        return _semantic_reference(
            x_ref,
            w1_ref,
            w2_ref,
            selected_experts,
            final_scales,
            intermediate_size,
            activation,
        )

    return reference


def _b12x_post_reference(
    x, w1, w2, selected_experts, final_scales, intermediate_size, view, activation
):
    # The b12x conformance tests intentionally use canonical BF16 weights as
    # the authority for both NVFP4 and checkpoint-style W4A16.
    return _semantic_reference(
        x, w1, w2, selected_experts, final_scales, intermediate_size, activation
    )


def _prepare_b12x_w4a16(BackendCfg, w1, w2, **kwargs):
    from flashinfer.fused_moe.prepare import _quantize_b12x_expert_weights

    q1, sf1 = _quantize_b12x_expert_weights(w1)
    q2, sf2 = _quantize_b12x_expert_weights(w2)
    ones = torch.ones(w1.shape[0], device=w1.device, dtype=torch.float32)
    return BackendCfg.prepare_weights(
        q1,
        sf1,
        ones,
        q2,
        sf2,
        ones.clone(),
        activation=kwargs["activation"],
        source_format="modelopt",
    )


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
        candidate_configs=(TrtllmBf16Config, CutlassBf16Config),
        snap=_bf16_snap,
        make_act_pack=_bf16_act_pack,
        make_act_pack_logits=_bf16_act_pack_logits,
        reference=_bf16_reference,
        poison=_poison_bf16_out,
        out_dtype=torch.bfloat16,
        atol_frac=0.05,  # initial; calibrate on SM100 (bf16 rounding floor)
        rtol=0.05,
    ),
    QuantVariant.DeepSeekFp8: DTypeHandler(
        variant=QuantVariant.DeepSeekFp8,
        candidate_configs=(TrtllmFp8BlockConfig,),
        snap=_block_fp8_snap,
        make_act_pack=lambda x, ids, weights: _block_fp8_act_pack(
            x, ids, weights, variant=QuantVariant.DeepSeekFp8
        ),
        make_act_pack_logits=lambda x, logits, bias: _block_fp8_act_pack_logits(
            x, logits, bias, variant=QuantVariant.DeepSeekFp8
        ),
        reference=lambda *args: _block_fp8_reference(
            *args, variant=QuantVariant.DeepSeekFp8
        ),
        poison=_poison_bf16_out,
        out_dtype=torch.bfloat16,
        atol_frac=0.15,  # provisional; recalibrate over the expanded SM100 sweep
        rtol=0.85,  # legacy-aligned initial bound, not a settled regression bar
    ),
    QuantVariant.MxFp8: DTypeHandler(
        variant=QuantVariant.MxFp8,
        candidate_configs=(TrtllmFp8BlockConfig,),
        snap=_block_fp8_snap,
        make_act_pack=lambda x, ids, weights: _block_fp8_act_pack(
            x, ids, weights, variant=QuantVariant.MxFp8
        ),
        make_act_pack_logits=lambda x, logits, bias: _block_fp8_act_pack_logits(
            x, logits, bias, variant=QuantVariant.MxFp8
        ),
        reference=lambda *args: _block_fp8_reference(*args, variant=QuantVariant.MxFp8),
        poison=_poison_bf16_out,
        out_dtype=torch.bfloat16,
        atol_frac=0.15,  # provisional; recalibrate over the expanded SM100 sweep
        rtol=0.85,  # legacy-aligned initial bound, not a settled regression bar
    ),
    QuantVariant.FP8PerTensor: DTypeHandler(
        variant=QuantVariant.FP8PerTensor,
        candidate_configs=(TrtllmFp8PerTensorConfig,),
        snap=_block_fp8_snap,
        make_act_pack=_fp8_per_tensor_act_pack,
        make_act_pack_logits=_fp8_per_tensor_act_pack_logits,
        reference=_fp8_per_tensor_reference,
        poison=_poison_bf16_out,
        out_dtype=torch.bfloat16,
        atol_frac=0.05,
        rtol=0.3,
    ),
    QuantVariant.MXFP4: DTypeHandler(
        variant=QuantVariant.MXFP4,
        candidate_configs=(TrtllmFp4Config,),
        snap=lambda t: _mxfp4_snap(t, bf16_activation=False),
        make_act_pack=lambda x, ids, weights: _mxfp4_act_pack(
            x, ids, weights, variant=QuantVariant.MXFP4
        ),
        make_act_pack_logits=lambda x, logits, bias: _mxfp4_act_pack_logits(
            x, logits, bias, variant=QuantVariant.MXFP4
        ),
        reference=lambda *args: _mxfp4_reference(*args, variant=QuantVariant.MXFP4),
        poison=_poison_bf16_out,
        out_dtype=torch.bfloat16,
        atol_frac=0.05,  # provisional; recalibrate over the expanded SM100 sweep
        rtol=0.3,
    ),
    QuantVariant.W4A16: DTypeHandler(
        variant=QuantVariant.W4A16,
        candidate_configs=(TrtllmFp4Config, CutlassW4A16Config),
        snap=lambda t: _mxfp4_snap(t, bf16_activation=True),
        make_act_pack=lambda x, ids, weights: _mxfp4_act_pack(
            x, ids, weights, variant=QuantVariant.W4A16
        ),
        make_act_pack_logits=lambda x, logits, bias: _mxfp4_act_pack_logits(
            x, logits, bias, variant=QuantVariant.W4A16
        ),
        reference=lambda *args: _mxfp4_reference(*args, variant=QuantVariant.W4A16),
        poison=_poison_bf16_out,
        out_dtype=torch.bfloat16,
        atol_frac=0.05,  # provisional; recalibrate over the expanded SM100 sweep
        rtol=0.3,
    ),
    QuantVariant.MxInt4: DTypeHandler(
        variant=QuantVariant.MxInt4,
        candidate_configs=(TrtllmMxInt4Config,),
        snap=_bf16_snap,
        make_act_pack=_bf16_act_pack,
        make_act_pack_logits=_mxint4_act_pack_logits,
        reference=_mxint4_reference,
        poison=_poison_bf16_out,
        out_dtype=torch.bfloat16,
        # A 160-seed GB200 (SM100) sweep exercised 17 MxInt4 cases; seed 18
        # requires atol_frac ~= 0.062 after the relative-tolerance contribution.
        # Keep a small SM100-calibrated margin; SM103/SM107 remain uncalibrated.
        atol_frac=0.065,
        rtol=0.3,
    ),
}


def _contract_handler(
    config_cls,
    variant,
    *,
    activation_pack,
    reference,
    snap=_block_fp8_snap,
    weight_snap=None,
    prepare_weights=None,
    atol_frac=0.2,
    rtol=0.2,
):
    return DTypeHandler(
        variant=variant,
        candidate_configs=(config_cls,),
        snap=snap,
        make_act_pack=activation_pack,
        make_act_pack_logits=None,
        reference=lambda *_: None,
        poison=_poison_bf16_out,
        out_dtype=torch.bfloat16,
        atol_frac=atol_frac,
        rtol=rtol,
        post_prepare_reference=reference,
        prepare_weights=prepare_weights,
        weight_snap=weight_snap,
    )


# Synthetic ids keep activation-pack contracts isolated from TRTLLM handlers.
# In particular, one Cfg creates exactly one activation pack, so backends with
# BF16, per-tensor FP8, and MXFP8 inputs cannot safely share a handler.
_CONTRACT_HANDLERS = {
    "cutlass_nvfp4": _contract_handler(
        CutlassNvfp4Config,
        QuantVariant.NVFP4,
        activation_pack=_contract_bf16_act_pack,
        reference=_cutlass_post_reference("cutlass_nvfp4"),
        snap=_snap_to_nvfp4,
        atol_frac=0.15,
        rtol=0.1,
    ),
    "cutlass_fp8_per_tensor": _contract_handler(
        CutlassFp8PerTensorConfig,
        QuantVariant.FP8PerTensor,
        activation_pack=_contract_fp8_act_pack(CutlassFp8PerTensorConfig),
        reference=_cutlass_post_reference("cutlass_fp8_per_tensor"),
        atol_frac=0.1,
        rtol=0.1,
    ),
    "cutlass_fp8_block": _contract_handler(
        CutlassFp8BlockConfig,
        QuantVariant.DeepSeekFp8,
        activation_pack=_contract_bf16_act_pack,
        reference=_cutlass_post_reference("cutlass_fp8_block"),
        atol_frac=0.1,
        rtol=0.1,
    ),
    "cutlass_mxfp8_mxfp4": _contract_handler(
        CutlassMxfp8Mxfp4Config,
        QuantVariant.MXFP4,
        activation_pack=_contract_fp8_act_pack(CutlassMxfp8Mxfp4Config),
        reference=_cutlass_post_reference("cutlass_mxfp8_mxfp4"),
        atol_frac=0.1,
        rtol=0.1,
    ),
    "cutlass_mxfp8": _contract_handler(
        CutlassMxfp8Config,
        QuantVariant.MxFp8,
        activation_pack=_contract_fp8_act_pack(CutlassMxfp8Config),
        reference=_cutlass_post_reference("cutlass_mxfp8"),
        atol_frac=0.1,
        rtol=0.1,
    ),
    "cutlass_w4a8": _contract_handler(
        CutlassW4A8Config,
        QuantVariant.W4A8,
        activation_pack=_contract_bf16_act_pack,
        reference=_cutlass_post_reference("cutlass_w4a8"),
        atol_frac=0.1,
        rtol=0.1,
    ),
    "cutlass_humming": _contract_handler(
        CutlassHummingConfig,
        QuantVariant.Humming,
        activation_pack=_contract_bf16_act_pack,
        reference=_cutlass_post_reference("cutlass_humming"),
    ),
    "b12x_nvfp4": _contract_handler(
        B12xNvfp4Config,
        QuantVariant.NVFP4,
        activation_pack=_contract_bf16_act_pack,
        reference=_b12x_post_reference,
        snap=_snap_to_nvfp4,
        atol_frac=0.15,
        rtol=0.1,
    ),
    "b12x_w4a16": _contract_handler(
        B12xW4A16Config,
        QuantVariant.W4A16,
        activation_pack=_contract_bf16_act_pack,
        reference=_b12x_post_reference,
        weight_snap=_snap_to_nvfp4,
        prepare_weights=_prepare_b12x_w4a16,
        # TODO: weight_snap removes the weight quantization error, so the
        # remaining tolerance covers kernel accumulation plus the
        # intermediate/epilogue rounding this FP32 oracle does not model.
        # Tighten once SM120/121 CI can calibrate it.
        atol_frac=0.05,
        rtol=0.3,
    ),
}
_B12X_BACKEND_KEYS = frozenset(("b12x_nvfp4", "b12x_w4a16"))
_FP8_BLOCK_BACKEND_KEY = "cutlass_fp8_block"

# Cfg.variant string <-> handler lookup (random-generation ids stay unchanged).
_HANDLER_BY_ID = {
    **{variant.name.lower(): handler for variant, handler in _DTYPE.items()},
    **_CONTRACT_HANDLERS,
}
_FROMLOGITS_VARIANT_IDS = tuple(
    variant.name.lower()
    for variant, handler in _DTYPE.items()
    if handler.make_act_pack_logits is not None
)
_PREROUTED_VARIANT_IDS = tuple(
    variant.name.lower()
    for variant, handler in _DTYPE.items()
    if handler.make_act_pack is not None
)


def _handler_for(cfg):
    return _HANDLER_BY_ID[cfg.variant]


def _activation_for(cfg):
    return {
        "swiglu": SwiGLU(),
        "geglu": GeGLU(),
        "situ": SiTU(),
        "relu2": ReLU2(),
        "geglutanh": GeGLUTanh(),
        "swiglustep": SwiGLUStep(),
        "identity": Identity(),
        "gelu": GELU(),
        "relu": ReLU(),
        "silu": SiLU(),
    }[cfg.activation]


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
# pre-routed host topk handles them trivially, but exercising all modes is free coverage.
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
    RoutingMethodType.TopKSigmoid,  # top_k(raw) -> sigmoid
    RoutingMethodType.DeepSeekV3,  # sigmoid+bias -> group-topk -> top_k (#2575 lives here)
    RoutingMethodType.MiniMax2,  # sigmoid+bias -> top_k -> scaled sum-norm
    RoutingMethodType.Llama4,  # top1 -> sigmoid (top_k forced to 1)
]
# Compiled in-kernel routing tiers are method-specific. Pre-routed modes bypass
# these limits, but FromLogits must not generate a shape that the selected
# routing policy cannot dispatch.
_FROMLOGITS_MAX_EXPERTS = {
    RoutingMethodType.Default: 256,
    RoutingMethodType.TopK: 256,
    RoutingMethodType.Sigmoid: 256,
    RoutingMethodType.SigmoidRenorm: 256,
    RoutingMethodType.Llama4: 128,
}
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
_UNPACKED_BACKENDS = {
    cfg_cls
    for cfg_cls, runner_cls in _BACKEND_RUNNERS.items()
    if RoutingInputMode.UnpackedPrecomputed in runner_cls.supported_routing_modes
}
# Backend config classes whose runner can compute an EP shard (a local expert subset with
# a nonzero offset), likewise derived from the runners' capability declaration. CUTLASS and
# b12x kernels compute the full routed set only, so an EP config restricts to these.
_EP_BACKENDS = {
    cfg_cls
    for cfg_cls, runner_cls in _BACKEND_RUNNERS.items()
    if runner_cls.supports_expert_parallelism
}
# Backend config classes whose runners support do_finalize=False and return the
# three-tensor unfinalized output contract.
_UNFINALIZED_BACKENDS = {
    cfg_cls
    for cfg_cls, runner_cls in _BACKEND_RUNNERS.items()
    if issubclass(runner_cls, _TrtllmRunnerBase)
}
_UNPACKED_VARIANT_IDS = tuple(
    variant.name.lower()
    for variant, handler in _DTYPE.items()
    if any(cfg_cls in _UNPACKED_BACKENDS for cfg_cls in handler.candidate_configs)
)

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
    # "prerouted" (PackedPrecomputed) | "unpacked" | "fromlogits"
    routing_input_mode: str = "prerouted"
    logits_dtype: str = "bf16"  # "bf16" | "fp32" (#2796 fp32-router-logits class)
    unpacked_weights_dtype: str = "bf16"  # "bf16" | "fp32"; unpacked mode only
    # Fused shared experts (S). Layer geometry, not a per-call value: the
    # weight tensors carry n_local + S rows and the routing kernel appends
    # ids [E, E+S) at weight 1.0 after the routed top-k.
    num_fused_shared_experts: int = 0
    n_group: int = 0  # DeepSeekV3 group count (0 -> None)
    topk_group: int = 0  # DeepSeekV3 groups kept (0 -> None)
    routed_scaling: float = 0.0  # DeepSeekV3 weight scale (0.0 -> None)
    do_finalize: bool = True
    activation: str = "swiglu"
    expected_backend: str = ""  # curated execution assertion; empty for random cases

    @property
    def n_weight_rows(self):  # physical expert-major rows: routed + shared
        return self.n_local + self.num_fused_shared_experts

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
    def is_unpacked(self):
        return self.routing_input_mode == "unpacked"

    @property
    def label(self):
        ep = f"L{self.n_local}o{self.expert_offset}_" if self.is_ep else ""
        mode = "FL_" if self.is_fromlogits else "UP_" if self.is_unpacked else ""
        ld = "fp32_" if self.logits_dtype == "fp32" else ""
        uwd = (
            "wfp32_"
            if self.is_unpacked and self.unpacked_weights_dtype == "fp32"
            else ""
        )
        grp = f"g{self.n_group}x{self.topk_group}_" if self.n_group else ""
        sh = (
            f"s{self.num_fused_shared_experts}_"
            if self.num_fused_shared_experts
            else ""
        )
        finalize = "" if self.do_finalize else "unfinalized_"
        return (
            f"{self.variant}_{self.activation}_{sh}{mode}{finalize}"
            f"{self.routing_method.name}_{ld}{uwd}{self.route}_"
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
    mode_roll = rng.random()
    fromlogits = mode_roll < 0.5
    unpacked = 0.5 <= mode_roll < 0.65
    force_non_ep = fromlogits or method == RoutingMethodType.DeepSeekV3
    # Resample shape until the weights of the FINAL config fit the budget (modest per-test
    # GPU footprint). Routing mode is chosen BEFORE this loop on purpose: non-EP-forced
    # configs (FromLogits / DeepSeekV3) hold the FULL expert set, so budgeting a sharded
    # `local` and flipping to non-EP afterwards would admit up to shards x the budget.
    eligible_experts = _EXPERTS
    if fromlogits and method in _FROMLOGITS_MAX_EXPERTS:
        max_experts = _FROMLOGITS_MAX_EXPERTS[method]
        eligible_experts = [ne for ne in _EXPERTS if ne <= max_experts]
    for _ in range(64):
        ne, h, i = (
            rng.choice(eligible_experts),
            rng.choice(_HIDDEN),
            rng.choice(_INTERMED),
        )
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
    else:
        raise RuntimeError(
            f"seed {seed} could not generate a unified-MoE fuzz shape within "
            f"the {_WEIGHT_ELEM_BUDGET}-element weight budget after 64 attempts"
        )

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

    fromlogits_variants = _FROMLOGITS_VARIANT_IDS
    prerouted_variants = _PREROUTED_VARIANT_IDS
    if method == RoutingMethodType.Llama4:
        # Per-tensor FP8 applies the Llama4 route scale on GEMM1 input rather
        # than in finalization, so it needs a method-aware reference.
        fromlogits_variants = tuple(
            variant for variant in fromlogits_variants if variant != "fp8pertensor"
        )
        prerouted_variants = tuple(
            variant for variant in prerouted_variants if variant != "fp8pertensor"
        )

    variant = (
        rng.choice(fromlogits_variants)
        if fromlogits
        else rng.choice(_UNPACKED_VARIANT_IDS)
        if unpacked
        else rng.choice(prerouted_variants)
    )
    # The legacy TRTLLM MXFP4 and MxInt4 modes are validated only with BF16 logits.
    logits_dtype = (
        "bf16"
        if variant in ("mxfp4", "w4a16", "mxint4")
        else ("fp32" if rng.random() < 0.25 else "bf16")
    )
    # Fused shared experts: only the DeepSeekV3 FromLogits path emits appended
    # slots. Block-FP8 and all TRTLLM FP4 variants forward S; EP is rejected.
    # Roll sparsely so this axis does not crowd out routed coverage.
    num_fused_shared_experts = 0
    if (
        method == RoutingMethodType.DeepSeekV3
        and fromlogits
        and offset == 0
        and local == ne
        and variant in ("deepseekfp8", "mxfp8", "nvfp4", "mxfp4", "w4a16")
        and rng.random() < 0.25
    ):
        # Bounded by the kernel ceilings on the FUSED totals.
        max_shared = min(2, 32 - top_k, 512 - ne)
        if (
            max_shared >= 1
            and _weight_elems(local + max_shared, h, i) <= _WEIGHT_ELEM_BUDGET
        ):
            num_fused_shared_experts = rng.randint(1, max_shared)

    activation = "swiglu"
    if variant == "bf16":
        # FromLogits and EP require TRTLLM, whose BF16 cubins expose SwiGLU
        # and ReLU2. CUTLASS-only activations are valid only for non-EP
        # pre-routed cases.
        activation = rng.choice(
            ("swiglu", "relu2")
            if fromlogits or offset != 0 or local != ne
            else ("swiglu", "relu2", "geglutanh", "swiglustep", "situ")
        )
    return Cfg(
        num_tokens=rng.choice(_TOKENS),
        hidden=h,
        intermediate=i,
        num_experts=ne,
        top_k=top_k,
        variant=variant,
        route=rng.choice(_ROUTE),
        seed=seed,
        local_experts=local,
        expert_offset=offset,
        routing_method=method,
        routing_input_mode=(
            "fromlogits" if fromlogits else "unpacked" if unpacked else "prerouted"
        ),
        logits_dtype=logits_dtype,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling=routed_scaling,
        activation=activation,
        num_fused_shared_experts=num_fused_shared_experts,
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
    Cfg(
        64,
        512,
        512,
        32,
        4,
        "bf16",
        "uniform",
        900_032,
        routing_method=RoutingMethodType.Default,
        routing_input_mode="fromlogits",
        logits_dtype="fp32",
    ),  # BF16 FromLogits with FP32 logits; seed % 4 == 0 exercises autotuning
    Cfg(
        64,
        512,
        512,
        32,
        4,
        "bf16",
        "uniform",
        900_033,
        routing_method=RoutingMethodType.DeepSeekV3,
        routing_input_mode="fromlogits",
        n_group=4,
        topk_group=2,
        routed_scaling=1.0,
    ),  # BF16 FromLogits bias/group routing
    # Deterministic typed-activation coverage. Keep CUTLASS-only activations
    # pre-routed so they have an executable candidate.
    Cfg(8, 256, 256, 8, 2, "bf16", "uniform", 900_014, activation="relu2"),
    Cfg(
        8,
        256,
        256,
        8,
        2,
        "bf16",
        "uniform",
        900_015,
        activation="geglutanh",
        expected_backend="cutlass_bf16",
    ),
    Cfg(
        8,
        256,
        256,
        8,
        2,
        "bf16",
        "uniform",
        900_022,
        activation="swiglustep",
        expected_backend="cutlass_bf16",
    ),
    Cfg(
        8,
        256,
        256,
        8,
        2,
        "bf16",
        "uniform",
        900_023,
        activation="situ",
        expected_backend="cutlass_bf16",
    ),
    Cfg(
        16,
        7168,
        2048,
        256,
        2,
        "deepseekfp8",
        "uniform",
        900_011,
        local_experts=2,
    ),  # DeepSeek-V3 dimensions with a two-expert local shard
    Cfg(
        256,
        1024,
        512,
        32,
        4,
        "deepseekfp8",
        "uniform",
        900_012,
        routing_method=RoutingMethodType.Default,
        routing_input_mode="fromlogits",
        logits_dtype="fp32",
    ),
    Cfg(256, 2048, 1024, 16, 4, "mxfp8", "imbalanced", 900_013),
    Cfg(
        256,
        1024,
        512,
        32,
        4,
        "mxfp8",
        "uniform",
        900_016,
        routing_method=RoutingMethodType.Default,
        routing_input_mode="fromlogits",
    ),  # seed % 4 == 0 deliberately exercises production autotuning for MXFP8
    Cfg(
        256,
        1024,
        512,
        32,
        4,
        "fp8pertensor",
        "uniform",
        900_020,
        routing_method=RoutingMethodType.Default,
        routing_input_mode="fromlogits",
        logits_dtype="fp32",
    ),  # per-tensor FP8 FromLogits; seed % 4 == 0 exercises autotuning
    Cfg(
        256,
        1024,
        512,
        32,
        4,
        "fp8pertensor",
        "uniform",
        900_036,
        routing_input_mode="prerouted",
    ),  # per-tensor FP8 packed routing; seed % 4 == 0 exercises autotuning
    Cfg(
        64,
        512,
        512,
        128,
        4,
        "nvfp4",
        "imbalanced",
        900_024,
        local_experts=32,
        expert_offset=32,
        routing_input_mode="unpacked",
    ),  # Unpacked BF16 weights + nonzero EP offset; seed % 4 == 0 autotunes
    Cfg(
        64,
        512,
        512,
        128,
        4,
        "nvfp4",
        "imbalanced",
        900_028,
        local_experts=32,
        expert_offset=32,
        routing_input_mode="unpacked",
        unpacked_weights_dtype="fp32",
    ),  # Unpacked FP32 weights + nonzero EP offset; seed % 4 == 0 autotunes
    Cfg(128, 1024, 512, 16, 4, "mxfp4", "uniform", 900_017),
    Cfg(128, 1024, 512, 16, 4, "w4a16", "imbalanced", 900_018),
    Cfg(
        128,
        1024,
        512,
        16,
        4,
        "mxfp4",
        "uniform",
        900_019,
        routing_method=RoutingMethodType.Default,
        routing_input_mode="fromlogits",
    ),
    Cfg(
        128,
        1024,
        512,
        16,
        4,
        "w4a16",
        "imbalanced",
        900_021,
        routing_method=RoutingMethodType.Default,
        routing_input_mode="fromlogits",
    ),
    Cfg(
        64,
        1024,
        512,
        16,
        4,
        "mxfp4",
        "uniform",
        900_042,
        routing_input_mode="unpacked",
    ),
    Cfg(
        64,
        1024,
        512,
        16,
        4,
        "w4a16",
        "imbalanced",
        900_043,
        routing_input_mode="unpacked",
        unpacked_weights_dtype="fp32",
    ),
    Cfg(
        64,
        512,
        512,
        16,
        4,
        "mxint4",
        "imbalanced",
        900_040,
    ),  # packed MxInt4; seed % 4 == 0 exercises production autotuning
    Cfg(
        64,
        512,
        512,
        16,
        4,
        "mxint4",
        "uniform",
        900_041,
        routing_method=RoutingMethodType.Default,
        routing_input_mode="fromlogits",
        logits_dtype="bf16",
    ),
    # Fused shared experts. Curated rather than left to the sparse random roll.
    Cfg(
        256,
        1024,
        512,
        128,
        8,
        "deepseekfp8",
        "uniform",
        900_037,
        routing_method=RoutingMethodType.DeepSeekV3,
        routing_input_mode="fromlogits",
        n_group=8,
        topk_group=4,
        routed_scaling=2.5,
        num_fused_shared_experts=1,
    ),
    Cfg(
        128,
        1024,
        512,
        128,
        8,
        "mxfp8",
        "imbalanced",
        900_038,
        routing_method=RoutingMethodType.DeepSeekV3,
        routing_input_mode="fromlogits",
        n_group=8,
        topk_group=4,
        routed_scaling=2.5,
        num_fused_shared_experts=2,
    ),
    *[
        Cfg(
            32,
            1024,
            512,
            32,
            8,
            variant,
            "uniform",
            seed,
            routing_method=RoutingMethodType.DeepSeekV3,
            routing_input_mode="fromlogits",
            logits_dtype="bf16",
            n_group=8,
            topk_group=4,
            routed_scaling=2.5,
            num_fused_shared_experts=num_shared,
        )
        for variant, num_shared, seed in (
            ("nvfp4", 1, 900_045),
            ("mxfp4", 2, 900_046),
            ("w4a16", 1, 900_044),
        )
    ],
    # Issue #3926: every unified TRTLLM operator that exposes unfinalized
    # intermediates must return BF16 expert weights for either logits dtype.
    *[
        Cfg(
            32,
            512,
            512,
            16,
            4,
            variant,
            "uniform",
            seed,
            routing_method=RoutingMethodType.Default,
            routing_input_mode="fromlogits",
            logits_dtype=logits_dtype,
            do_finalize=False,
        )
        for variant, seed_base in (
            ("bf16", 900_050),
            ("fp8pertensor", 900_052),
            ("deepseekfp8", 900_054),
            ("mxint4", 900_056),
            ("nvfp4", 900_058),
            ("mxfp8", 900_060),
        )
        for logits_dtype, seed in (
            ("bf16", seed_base),
            ("fp32", seed_base + 1),
        )
    ],
    # PackedPrecomputed + unfinalized coverage. Packed ids encode BF16 routing
    # weights, but FP4 borrows topk_weights instead of allocating its own buffer.
    # The value assertion below guards the returned weights across all TRTLLM
    # variants and catches an invalid FP4 buffer.
    *[
        Cfg(
            32,
            512,
            512,
            16,
            4,
            variant,
            "uniform",
            seed,
            routing_method=RoutingMethodType.RenormalizeNaive,
            routing_input_mode="prerouted",
            do_finalize=False,
        )
        for variant, seed in (
            ("bf16", 900_070),
            ("fp8pertensor", 900_071),
            ("deepseekfp8", 900_072),
            ("mxint4", 900_073),
            ("nvfp4", 900_074),
            ("mxfp8", 900_075),
        )
    ],
    # Contract-isolated CUTLASS and b12x coverage. These are deliberately
    # curated-only: random generation and its historical seed stream remain
    # unchanged. Every case is PackedPrecomputed, finalized, non-EP, and has a
    # singleton backend candidate through its synthetic handler id.
    *[
        Cfg(
            16,
            128,
            256,
            4,
            2,
            variant,
            "uniform",
            seed,
            activation=activation,
            expected_backend=backend,
        )
        for variant, backend, activation, seed in (
            ("cutlass_nvfp4", "cutlass_nvfp4", "swiglu", 900_080),
            ("cutlass_nvfp4", "cutlass_nvfp4", "geglutanh", 900_081),
            (
                "cutlass_fp8_per_tensor",
                "cutlass_fp8_per_tensor",
                "swiglu",
                900_082,
            ),
            (
                "cutlass_fp8_per_tensor",
                "cutlass_fp8_per_tensor",
                "geglutanh",
                900_083,
            ),
            ("cutlass_fp8_block", "cutlass_fp8_block", "swiglu", 900_084),
            ("cutlass_fp8_block", "cutlass_fp8_block", "geglutanh", 900_085),
            (
                "cutlass_mxfp8_mxfp4",
                "cutlass_mxfp8_mxfp4",
                "swiglu",
                900_086,
            ),
            (
                "cutlass_mxfp8_mxfp4",
                "cutlass_mxfp8_mxfp4",
                "geglutanh",
                900_087,
            ),
            ("cutlass_mxfp8", "cutlass_mxfp8", "swiglu", 900_088),
            ("cutlass_mxfp8", "cutlass_mxfp8", "geglutanh", 900_089),
            ("cutlass_w4a8", "cutlass_w4a8", "swiglu", 900_090),
            ("cutlass_w4a8", "cutlass_w4a8", "geglutanh", 900_091),
            ("cutlass_humming", "cutlass_humming", "swiglu", 900_092),
            ("cutlass_humming", "cutlass_humming", "geglutanh", 900_093),
            ("b12x_nvfp4", "b12x_nvfp4", "swiglu", 900_094),
            ("b12x_nvfp4", "b12x_nvfp4", "geglutanh", 900_095),
            ("b12x_w4a16", "b12x_w4a16", "swiglu", 900_096),
            ("b12x_w4a16", "b12x_w4a16", "relu2", 900_097),
            # Non-gated geometry (gemm1_rows == I, no up/gate split). CUTLASS
            # NVFP4 declares all four, so these reach the shared
            # _apply_typed_activation branches that the gated cases cannot.
            ("cutlass_nvfp4", "cutlass_nvfp4", "identity", 900_098),
            ("cutlass_nvfp4", "cutlass_nvfp4", "gelu", 900_099),
            ("cutlass_nvfp4", "cutlass_nvfp4", "relu", 900_100),
            ("cutlass_nvfp4", "cutlass_nvfp4", "silu", 900_101),
        )
    ],
]
_CURATED_BY_SEED = {}
for _cfg in _CURATED:
    if _cfg.seed in _CURATED_BY_SEED:
        raise ValueError(
            "duplicate curated unified-MoE fuzz seed "
            f"{_cfg.seed}: {_CURATED_BY_SEED[_cfg.seed].label} and {_cfg.label}"
        )
    _CURATED_BY_SEED[_cfg.seed] = _cfg


def test_random_seed_stream_is_unchanged():
    """Lock the raw ``_gen(i)`` stream independently of curated overlays."""
    payload = "\n".join(repr(_gen(i)) for i in range(160)).encode()
    assert (
        hashlib.sha256(payload).hexdigest()
        == "cff06d91b524c74c66863e4078e24303b431eeffbedc94b3237390bccd837e70"
    )


def test_contract_handler_inventory_is_single_backend_and_non_deterministic():
    expected = {
        "cutlass_nvfp4": "cutlass_nvfp4",
        "cutlass_fp8_per_tensor": "cutlass_fp8_per_tensor",
        "cutlass_fp8_block": "cutlass_fp8_block",
        "cutlass_mxfp8_mxfp4": "cutlass_mxfp8_mxfp4",
        "cutlass_mxfp8": "cutlass_mxfp8",
        "cutlass_w4a8": "cutlass_w4a8",
        "cutlass_humming": "cutlass_humming",
        "b12x_nvfp4": "b12x_nvfp4",
        "b12x_w4a16": "b12x_w4a16",
    }
    for handler_id, backend_key in expected.items():
        handler = _HANDLER_BY_ID[handler_id]
        assert len(handler.candidate_configs) == 1
        config_type = handler.candidate_configs[0]
        assert _BACKEND_RUNNERS[config_type].backend_key == backend_key
        assert backend_key not in _DETERMINISTIC


def test_b12x_w4a16_uses_nvfp4_weight_snap():
    # Its reference treats the canonical BF16 weights as the authority, so they
    # must already sit on the grid _quantize_b12x_expert_weights will use.
    assert _HANDLER_BY_ID["b12x_w4a16"].weight_snap is _snap_to_nvfp4


def test_contract_curated_seeds_match_declared_capabilities():
    contract_cases = [cfg for cfg in _CURATED if cfg.variant in _CONTRACT_HANDLERS]
    assert {cfg.seed for cfg in contract_cases} == set(range(900_080, 900_102))
    for cfg in contract_cases:
        handler = _handler_for(cfg)
        config_type = handler.candidate_configs[0]
        runner_type = _BACKEND_RUNNERS[config_type]
        assert handler.variant in runner_type.supported_quant_variants
        by_quant = runner_type.supported_activation_classes_by_quant
        activations = (
            by_quant[handler.variant]
            if by_quant
            else runner_type.supported_activation_classes
        )
        assert type(_activation_for(cfg)) in activations
        assert cfg.routing_input_mode == "prerouted"
        assert cfg.do_finalize and not cfg.is_ep
        assert cfg.num_fused_shared_experts == 0
        assert runner_type.backend_key == cfg.expected_backend


def test_semantic_reference_applies_situ_clamp():
    activation = SiTU(gate_scale=4.0, linear_scale=25.0, clamp_limit=1.0)
    x = torch.tensor([[2.0]])
    w1 = torch.tensor([[[3.0], [4.0]]])
    w2 = torch.ones(1, 1, 1)
    selected = torch.zeros(1, 1, dtype=torch.int32)
    routing_weights = torch.ones(1, 1)

    actual = _semantic_reference(x, w1, w2, selected, routing_weights, 1, activation)
    up = torch.tensor(1.0)
    gate = torch.tensor(1.0)
    expected = (
        activation.linear_scale
        * torch.tanh(up / activation.linear_scale)
        * activation.gate_scale
        * torch.tanh(gate / activation.gate_scale)
        * torch.sigmoid(gate)
    )
    torch.testing.assert_close(actual, expected.reshape(1, 1))


if _ONLY_SEEDS:  # perfect-repro: run only the named seed(s)
    _CONFIGS = [
        _CURATED_BY_SEED.get(s) or _gen(s)
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
    elif method == M.TopKSigmoid:  # top_k(raw) -> sigmoid over selected
        raw, sel = torch.topk(lf, top_k, dim=-1)
        w = torch.sigmoid(raw)
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
            gmask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(
                -1, gidx, True
            )
            smask = (
                gmask.unsqueeze(-1)
                .expand(*sel_scores.shape[:-1], n_group, E // n_group)
                .reshape(sel_scores.shape)
            )
            # A routing bias can make scores negative. Zero-masking would let an
            # unselected expert outrank a valid negative score in the selected group.
            sel_scores = sel_scores.masked_fill(~smask, float("-inf"))
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


def test_deepseek_v3_route_excludes_unselected_groups_with_negative_scores():
    logits = torch.zeros((1, 8), dtype=torch.float32)
    # Group 0 wins by its top-2 sum, but its fourth selection score is negative.
    # Experts in group 1 must remain ineligible rather than becoming zero-score
    # candidates that can displace that valid negative-score expert.
    bias = torch.tensor(
        [[2.5, 1.5, 0.5, -1.5, 0.0, -0.1, -0.2, -0.3]],
        dtype=torch.float32,
    )

    selected, _ = _route(
        logits,
        RoutingMethodType.DeepSeekV3,
        top_k=4,
        bias=bias,
        n_group=2,
        topk_group=1,
        routed_scaling=1.0,
    )

    assert set(selected[0].tolist()) == {0, 1, 2, 3}


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

    def sparse(*shape, snap=handler.snap):
        dense = torch.randn(*shape, device="cuda", generator=g)
        keep = torch.rand(shape, device="cuda", generator=g) >= 0.75  # ~75% zeros
        return snap(dense * keep)

    # Expert-major tensors carry the SHARED rows too (routed first, shared
    # appended); E_local stays routed-only, matching the API contract.
    rows = cfg.n_weight_rows
    gemm1_rows = 2 * I if _activation_for(cfg).is_gated else I
    x = sparse(T, H)
    weight_snap = handler.weight_snap or handler.snap
    w1 = sparse(rows, gemm1_rows, H, snap=weight_snap)
    w2 = sparse(rows, H, I, snap=weight_snap)

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

    # Mirror the routing kernel's shared-expert append: ids [E, E+S) at weight
    # exactly 1.0, after the routed top-k. The reference sums over these the
    # same way it does the routed slots, so no reference change is needed --
    # it already resolves a global id to a weight row via expert_offset and
    # sizes itself from w1.shape[0].
    S = cfg.num_fused_shared_experts
    if S:
        shared_ids = torch.arange(
            cfg.num_experts, cfg.num_experts + S, device="cuda", dtype=torch.int32
        ).expand(T, S)
        selected_experts = torch.cat((selected_experts, shared_ids), dim=1)
        final_scales = torch.cat(
            (
                final_scales,
                torch.ones((T, S), device="cuda", dtype=final_scales.dtype),
            ),
            dim=1,
        )
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
# Broad skip terms also appear in real activation-plumbing errors, so these
# concrete identifiers force such errors to fail. Do not add activation names:
# "does not support SiTU" is a legitimate capability skip.
_NEVER_SKIP_SUBSTR = (
    "activation parameters",
    "gemm1_alpha",
    "gemm1_beta",
    "gemm1_clamp_limit",
)


def _is_unsupported(e):
    msg = str(e).lower()
    if any(c in msg for c in _CRASH_SUBSTR):
        return False  # a crash is always a finding, never "unsupported"
    # A runner raises NotImplementedError only for declared capability gaps, so
    # honor it before the substring heuristics can second-guess the message.
    if isinstance(e, NotImplementedError):
        return True
    if any(s in msg for s in _NEVER_SKIP_SUBSTR):
        return False
    return any(s in msg for s in _SKIP_SUBSTR)


# Fragments, not full messages: the classifier must also match what a runner
# raises on its own, whose prefix need not match the preflight wording. Keeping
# the fragment primitive makes it a substring of the preflight reason by
# construction, so the two cannot drift apart.
_ENV_NEEDS_CUDA13 = "requires CUDA 13 or later"
_ENV_NEEDS_CUTE_DSL = "requires the CuTe DSL package"
_ENV_NEEDS_CUDA128 = "FP8 block scaling requires CUDA 12.8 or newer"
_CONTRACT_ENVIRONMENT_ERRORS = tuple(
    fragment.lower()
    for fragment in (_ENV_NEEDS_CUDA13, _ENV_NEEDS_CUTE_DSL, _ENV_NEEDS_CUDA128)
)


def _is_contract_environment_unavailable(e: Exception) -> bool:
    # Deliberately fail closed: only known environment diagnostics may skip.
    # A reworded or new rejection should fail CI until it is classified rather
    # than silently masking a backend capability regression.
    msg = str(e).lower()
    return any(reason in msg for reason in _CONTRACT_ENVIRONMENT_ERRORS)


@functools.cache
def _cuda_toolkit_version() -> tuple[int, int]:
    version = get_cuda_version()
    return version.major, version.minor


def _contract_preflight_skip_reason(
    cfg: Cfg,
    *,
    cuda_version: tuple[int, int] | None = None,
    cute_dsl_available: bool | None = None,
) -> str | None:
    if cfg.variant in _B12X_BACKEND_KEYS:
        if cuda_version is None:
            cuda_version = _cuda_toolkit_version()
        if cuda_version < (13, 0):
            return f"b12x unified MoE {_ENV_NEEDS_CUDA13}"
        if cute_dsl_available is None:
            cute_dsl_available = is_cute_dsl_available()
        if not cute_dsl_available:
            return f"b12x unified MoE {_ENV_NEEDS_CUTE_DSL}"
    elif cfg.variant == _FP8_BLOCK_BACKEND_KEY:
        if cuda_version is None:
            cuda_version = _cuda_toolkit_version()
        if cuda_version < (12, 8):
            return _ENV_NEEDS_CUDA128
    return None


@pytest.mark.parametrize(
    "exc,expected",
    (
        # A declared capability gap is a skip, even when the message names an
        # activation -- matching on activation names would misreport this.
        (NotImplementedError("backend does not support SiTU"), True),
        (NotImplementedError("TrtllmBf16RoutedRunner does not support GeGLU"), True),
        # Activation plumbing phrased as a shape error is the regression this
        # fuzzer exists to catch, despite "must be" being a skip term.
        (ValueError("gemm1_alpha must have shape (8,)"), False),
        (
            ValueError("runner: prepared weights are missing activation parameters"),
            False,
        ),
        # Ordinary shape-capability rejections still skip.
        (ValueError("hidden_size must be divisible by 128"), True),
        # A crash is never "unsupported", however it is phrased.
        (RuntimeError("CUDA error: illegal memory access"), False),
    ),
)
def test_is_unsupported_classification(exc, expected):
    assert _is_unsupported(exc) is expected


@pytest.mark.parametrize(
    "exc,expected",
    (
        (ValueError("b12x unified MoE requires CUDA 13 or later."), True),
        (RuntimeError("b12x unified MoE requires the CuTe DSL package."), True),
        (NotImplementedError("backend does not support SiTU"), False),
        (ValueError("hidden_size must be divisible by 128"), False),
    ),
)
def test_contract_environment_classification(exc, expected):
    assert _is_contract_environment_unavailable(exc) is expected


def test_b12x_contract_preflight_requires_cuda_13():
    b12x = _CURATED_BY_SEED[900_094]
    cutlass = _CURATED_BY_SEED[900_080]
    assert (
        _contract_preflight_skip_reason(
            b12x, cuda_version=(12, 9), cute_dsl_available=True
        )
        is not None
    )
    assert (
        _contract_preflight_skip_reason(
            b12x, cuda_version=(13, 0), cute_dsl_available=True
        )
        is None
    )
    assert _contract_preflight_skip_reason(cutlass, cuda_version=(12, 7)) is None


def test_contract_preflight_checks_backend_environment():
    b12x = _CURATED_BY_SEED[900_094]
    fp8_block = _CURATED_BY_SEED[900_084]
    assert (
        _contract_preflight_skip_reason(
            b12x, cuda_version=(13, 0), cute_dsl_available=False
        )
        is not None
    )
    assert _contract_preflight_skip_reason(fp8_block, cuda_version=(12, 7)) is not None
    assert _contract_preflight_skip_reason(fp8_block, cuda_version=(12, 8)) is None


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


@pytest.mark.shard_group("unified-moe-accumulated")
@pytest.mark.parametrize("cfg", _CONFIGS, ids=[c.label for c in _CONFIGS])
def test_unified_moe_fuzz(cfg):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    # Crash-class quarantine gate: MUST precede any kernel launch (a quarantined config would
    # poison the CUDA context for every later test in this process).
    LEDGER.xfail_if_quarantined(cfg)
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
    if reason := _contract_preflight_skip_reason(cfg):
        pytest.skip(reason)
    if handler.variant is QuantVariant.W4A16 and sm == 103:
        pytest.skip("TRTLLM MXFP4×BF16 is disabled on SM103")
    if handler.variant is QuantVariant.MxInt4 and (
        cfg.hidden % 256 != 0 or cfg.intermediate % 256 != 0
    ):
        pytest.skip("TRTLLM MxInt4 requires hidden/intermediate divisible by 256")
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
    elif cfg.is_unpacked:
        # Exact TRTLLM Mode 3 is currently wired only through the FP4 runner.
        # Backends that accept separate tensors through their own ABI are not
        # implementations of RoutingInputMode.UnpackedPrecomputed.
        wired_backends = [B for B in wired_backends if B in _UNPACKED_BACKENDS]
    if not cfg.do_finalize:
        # Only TRTLLM returns unfinalized intermediates; like the EP filter
        # below, this keeps an unsupported arch on the precise skip path.
        wired_backends = [B for B in wired_backends if B in _UNFINALIZED_BACKENDS]
    if cfg.is_ep:
        # An EP shard needs the runner to map global ids onto a local expert subset;
        # backends without that capability (CUTLASS, b12x) would fail MoELayer's
        # check_support, and with no other candidate the case must SKIP, not FAIL.
        wired_backends = [B for B in wired_backends if B in _EP_BACKENDS]
    if _BACKEND_FILTER:
        wired_backends = [
            B
            for B in wired_backends
            if _BACKEND_RUNNERS[B].backend_key in _BACKEND_FILTER
        ]
    expected_backend_available = bool(
        cfg.expected_backend
        and any(
            _BACKEND_RUNNERS[B].backend_key == cfg.expected_backend and B.supported(sm)
            for B in handler.candidate_configs
            if B in _BACKEND_RUNNERS
        )
    )
    # The debug allowlist deliberately narrows execution to one backend, so it
    # must retire the curated assertion rather than trip it: otherwise bisecting
    # with FLASHINFER_UMOE_FUZZ_BACKENDS fails every curated case by design.
    if _BACKEND_FILTER and cfg.expected_backend not in _BACKEND_FILTER:
        expected_backend_available = False
    if expected_backend_available and not any(
        _BACKEND_RUNNERS[B].backend_key == cfg.expected_backend for B in wired_backends
    ):
        pytest.fail(
            f"{cfg.label}: expected backend {cfg.expected_backend!r} was filtered "
            "before execution"
        )
    # A backend-scoped crash quarantine must take effect before backend-native
    # weight preparation or MoELayer construction: both can load modules and
    # launch CUDA preparation kernels. Keep the findings so the overall case
    # still reports XFAIL after all healthy backends have run.
    quarantined_backends = []
    healthy_backends = []
    for BackendCfg in wired_backends:
        backend_key = _BACKEND_RUNNERS[BackendCfg].backend_key
        quarantine = LEDGER.skip_backend(cfg, backend_key)
        if quarantine:
            quarantined_backends.append((quarantine, backend_key))
            # A quarantined backend is intentionally never launched, so the
            # curated "expected backend must execute" assertion cannot hold.
            # Leaving it armed would report a ledger-tracked crash as a plain
            # test failure instead of the XFAIL the ledger asks for.
            if backend_key == cfg.expected_backend:
                expected_backend_available = False
        else:
            healthy_backends.append(BackendCfg)
    wired_backends = healthy_backends
    if not wired_backends:
        LEDGER.report_expected_failures(
            quarantined_backends,
            context=f"all candidate backends quarantined for {cfg.label}",
        )
        mode = (
            "in-kernel-routing "
            if cfg.is_fromlogits
            else "unpacked-precomputed "
            if cfg.is_unpacked
            else ""
        )
        pytest.skip(f"no wired {mode}backend for {cfg.variant} on SM{sm}")

    x, w1, w2, selected_experts, final_scales, logits, routing_bias = _master(
        cfg, handler
    )
    unpacked_weights_dtype = (
        torch.float32 if cfg.unpacked_weights_dtype == "fp32" else torch.bfloat16
    )
    reference_scales = (
        final_scales.to(unpacked_weights_dtype).float()
        if cfg.is_unpacked
        else final_scales
    )
    reference_args = (
        x,
        w1,
        w2,
        selected_experts,
        reference_scales,
        cfg.intermediate,
        cfg.expert_offset,
    )
    ref = None
    if handler.post_prepare_reference is None:
        if handler.variant is QuantVariant.BF16:
            ref = handler.reference(*reference_args, activation=_activation_for(cfg))
        else:
            ref = handler.reference(*reference_args)

    # One activation pack + one weight pack with each backend's native view, all built from the
    # SAME bf16 inputs (this rank's LOCAL shard) via the API's uniform prepare_weights. In-kernel
    # routing hands the kernel raw logits (+ bias); pre-routed hands it the host selection.
    if cfg.is_fromlogits:
        assert handler.make_act_pack_logits is not None
        act_pack = handler.make_act_pack_logits(x, logits, routing_bias)
    else:
        assert handler.make_act_pack is not None
        act_pack = handler.make_act_pack(x, selected_experts, final_scales)
        if cfg.is_unpacked:
            act_pack = MoEActivationPack(
                hidden_states_q=act_pack.hidden_states_q,
                hidden_states_scale=act_pack.hidden_states_scale,
                topk_ids=selected_experts,
                topk_weights=final_scales.to(unpacked_weights_dtype),
                routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
            )
    weight_pack = MoEWeightPack()
    prepared_views = {}
    for BackendCfg in wired_backends:
        prepare_kwargs = dict(
            num_local_experts=cfg.n_weight_rows,
            hidden_size=cfg.hidden,
            intermediate_size=cfg.intermediate,
            device=dev,
            activation=_activation_for(cfg),
        )
        # FP8BlockConfig distinguishes DeepSeekFp8/MxFp8; FP4Config distinguishes
        # NVFP4/MXFP4/W4A16. Both need the logical variant to select preparation.
        if BackendCfg in (TrtllmFp8BlockConfig, TrtllmFp4Config):
            prepare_kwargs["variant"] = handler.variant
        elif BackendCfg is TrtllmFp8PerTensorConfig:
            prepare_kwargs.update(
                hidden_states_scale_global=_fp8_per_tensor_global_scale(x),
                intermediate_scale_global=torch.tensor(64.0, device=dev),
            )
        view = (
            handler.prepare_weights(BackendCfg, w1, w2, **prepare_kwargs)
            if handler.prepare_weights is not None
            else BackendCfg.prepare_weights(w1, w2, **prepare_kwargs)
        )
        backend_key = _BACKEND_RUNNERS[BackendCfg].backend_key
        prepared_views[backend_key] = view
        weight_pack.prepare_for(backend_key, view)

    if handler.post_prepare_reference is not None:
        assert len(prepared_views) == 1
        view = next(iter(prepared_views.values()))
        # Keep activation quantization artifacts beside the prepared view only
        # for the independent reference; they are not added to MoEWeightPack.
        reference_view = dict(view)
        reference_view["_activation_q"] = act_pack.hidden_states_q
        reference_view["_activation_scale"] = act_pack.hidden_states_scale
        ref = handler.post_prepare_reference(
            x,
            w1,
            w2,
            selected_experts,
            final_scales,
            cfg.intermediate,
            reference_view,
            _activation_for(cfg),
        )
    assert ref is not None
    ref_abs_max = ref.abs().max().item()
    atol = handler.atol_frac * ref_abs_max + 1e-3
    rtol = handler.rtol

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
            num_fused_shared_experts=cfg.num_fused_shared_experts,
        ),
        activation=_activation_for(cfg),
        backend=BackendOptions(
            candidates=tuple(BackendCfg() for BackendCfg in wired_backends)
        ),
        execution=ExecutionConfig(
            tune_max_num_tokens=(
                cfg.num_tokens
                if os.environ.get("FLASHINFER_UMOE_FUZZ_TUNE_REAL_SHAPE")
                else max(cfg.num_tokens, 8192)
            )
        ),
        finalize=MoEFinalizeConfig(do_finalize=cfg.do_finalize),
    )

    try:
        layer = MoELayer(config)
    except Exception as e:
        if _is_unsupported(e):
            if (
                cfg.variant in _CONTRACT_HANDLERS
                and _is_contract_environment_unavailable(e)
            ):
                pytest.skip(
                    f"contract backend {cfg.expected_backend!r} unavailable: {e}"
                )
            if expected_backend_available:
                pytest.fail(
                    f"{cfg.label}: expected backend {cfg.expected_backend!r} "
                    f"failed MoELayer construction: {e}"
                )
            pytest.skip(f"MoELayer rejected {cfg.label}: {e}")
        raise

    if not cfg.do_finalize:
        result = layer(act_pack, weight_pack)
        assert isinstance(result, list) and len(result) == 3, (
            "do_finalize=False must return "
            "[gemm2_output, expert_weights, expanded_idx_to_permuted_idx], got "
            f"{type(result).__name__} of length "
            f"{len(result) if isinstance(result, list) else 'n/a'}"
        )
        gemm2_output, expert_weights, expanded_idx_to_permuted_idx = result
        assert gemm2_output.shape[-1] == cfg.hidden
        assert expert_weights.dtype == torch.bfloat16, (
            "do_finalize=False expert_weights must be bfloat16, got "
            f"{expert_weights.dtype} for routing_logits dtype {cfg.logits_dtype}"
        )
        assert tuple(expert_weights.shape) == (cfg.num_tokens, cfg.top_k)
        assert expanded_idx_to_permuted_idx.numel() == cfg.num_tokens * cfg.top_k
        torch.testing.assert_close(
            expert_weights.float(),
            final_scales.to(torch.bfloat16).float(),
            rtol=2e-2,
            atol=2e-3,
        )

        permutation = expanded_idx_to_permuted_idx.to(torch.long)
        assert (permutation >= 0).all()
        assert permutation.unique().numel() == permutation.numel()
        assert permutation.max().item() < gemm2_output.shape[0]

        recombined = (
            gemm2_output[permutation]
            .view(cfg.num_tokens, cfg.top_k, cfg.hidden)
            .float()
            .mul(expert_weights.float().unsqueeze(-1))
            .sum(dim=1)
        )
        abs_diff = (recombined - ref).abs()
        over_tol = abs_diff > (atol + rtol * ref.abs())
        if over_tol.any():
            _fail(
                cfg,
                "unfinalized host recombination",
                f"{int(over_tol.sum())}/{recombined.numel()} elems exceed tol",
                recombined,
                ref,
            )
        return

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
        # b12x's wrapper allocates and returns its output internally, so
        # pack_inputs intentionally has no caller-visible output to initialize.
        # Its clean run still receives the numeric and non-finite checks below.
        if runner.backend_key not in _B12X_BACKEND_KEYS:
            assert bufs, "could not locate the output buffer in pack_inputs"
        for b in bufs:
            handler.poison(b, poison_gen) if poison else b.zero_()
        out = runner.forward(inputs, tactic=-1)
        out = (out[0] if isinstance(out, (list, tuple)) else out).float()
        torch.cuda.synchronize()
        return out

    def assert_correct(out, tag):
        if ref_abs_max > 0 and out.abs().max().item() == 0:
            _fail(
                cfg,
                tag,
                "all-zero output for a nonzero reference",
                out,
                ref,
            )
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
        if os.environ.get("FLASHINFER_UMOE_FUZZ_LEAN"):
            return  # debug: minimal per-config work for sanitizer runs (gh #3957)
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
        # b12x cannot participate because its wrapper does not expose that
        # internally allocated output through pack_inputs.
        if runner.backend_key not in _B12X_BACKEND_KEYS:
            assert_correct(run(runner, poison=True), f"{tag} [poisoned-output]")
        # (5) autotune-tactic sweep: EVERY valid tactic must be correct, not just the default --
        # the autotuner-picks-a-bad-tactic class (#3168/#3227) on the real MoELayer dispatch.
        inputs = runner.pack_inputs(act_pack, weight_pack)
        try:
            tactics = runner.get_valid_tactics(inputs, None)
        except Exception:
            if cfg.variant in _CONTRACT_HANDLERS:
                raise
            tactics = []  # backend needs a profile object -> skip the sweep (default tactic stands)
        for tactic in tactics:
            o = runner.forward(inputs, tactic=tactic)
            o = (o[0] if isinstance(o, (list, tuple)) else o).float()
            torch.cuda.synchronize()
            assert_correct(o, f"{tag} [tactic={tactic}]")

    n_ran = 0
    ran_backends = set()
    expected_failures = []
    for runner in layer.runners:
        try:
            out = run(runner)
        except Exception as e:
            if _is_unsupported(e):
                continue  # backend rejects this shape -> skip; a crash re-raises
            raise
        tag = f"{runner.backend_key} {cfg.label}"
        n_ran += 1
        ran_backends.add(runner.backend_key)

        known = LEDGER.find(cfg, backend=runner.backend_key)
        if known:  # tracked bug -> run it, tolerate a wrong answer, but flag if it starts passing
            try:
                check_backend(runner, out, tag)
            except (AssertionError, pytest.fail.Exception):
                expected_failures.append((known, tag))
                continue
            LEDGER.flag_xpass(known, tag)
        else:
            check_backend(runner, out, tag)

    if n_ran == 0:
        LEDGER.report_expected_failures(
            quarantined_backends,
            context=f"all candidate backends quarantined for {cfg.label}",
        )
        pytest.skip(f"no runner ran {cfg.label} on SM{sm}")
    if expected_backend_available and cfg.expected_backend not in ran_backends:
        pytest.fail(
            f"{cfg.label}: expected backend {cfg.expected_backend!r} did not execute"
        )

    # (6) autotune-ON: drive the REAL production path -- MoELayer._select_winner profiles every
    # tactic of every runner (the #3168 profiling-IMA class) then selects + caches a winner; the
    # autotuned output must match the authoritative reference. Gated to a subset (profiling is slow)
    # and skipped if a candidate has a known failure (the tuner could pick the broken backend).
    autotune_due = (
        not _NO_AUTOTUNE
        and cfg.seed % 4 == 0
        and not any(
            LEDGER.find(cfg, backend=_BACKEND_RUNNERS[B].backend_key)
            for B in wired_backends
        )
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
    LEDGER.report_expected_failures(
        [*expected_failures, *quarantined_backends],
        context=f"tracked backend failures for {cfg.label}",
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
_CACHE_VARIANTS = (
    QuantVariant.NVFP4,
    QuantVariant.BF16,
)  # The 21-tactic block-FP8 runners make this multi-bucket stress sweep prohibitive.


@pytest.mark.parametrize(
    "variant", _CACHE_VARIANTS, ids=[v.name.lower() for v in _CACHE_VARIANTS]
)
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
        prepare_kwargs = dict(
            num_local_experts=E,
            hidden_size=H,
            intermediate_size=I,
            device=dev,
        )
        # FP8BlockConfig distinguishes DeepSeekFp8/MxFp8; FP4Config distinguishes
        # NVFP4/MXFP4/W4A16. Both need the logical variant to select preparation.
        if B in (TrtllmFp8BlockConfig, TrtllmFp4Config):
            prepare_kwargs["variant"] = variant
        weight_pack.prepare_for(
            _BACKEND_RUNNERS[B].backend_key,
            B.prepare_weights(
                w1,
                w2,
                **prepare_kwargs,
            ),
        )
    layer = MoELayer(
        MoEConfig(
            routing=RoutingConfig(num_experts=E, top_k=top_k),
            quant=QuantConfig(variant=variant),
            experts=ExpertConfig(intermediate_size=I, local_num_experts=E),
            activation=SwiGLU(),
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
