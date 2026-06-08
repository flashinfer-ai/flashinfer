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

A small ``_KNOWN_FAILURES`` ledger xfails already-filed bugs (e.g. trtllm EP offset>0 -> all-zero,
EP_OFFSET_FINDING.md): the case is still *run* so the suite stays green on a tracked bug yet flags
loudly the day it starts passing (fixed). A crash is never tolerated -- only a wrong answer.

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
  6. **device-state probe** after each config: a context-corrupting IMA surfaces as a failed
     alloc/launch or non-finite probe, turning silent corruption into a clean failure.

(Cross-backend agreement is intentionally NOT a check: with an authoritative tight reference, a
deviating backend is caught by #2 directly, and #2 also names which backend -- so a cross-backend
comparison adds no pass/fail power, only redundancy. See the design discussion.)

Coverage today: NVFP4 (CuteDSL + TRTLLM-FP4-routed) on SM100 -- the only wired MVP runners.
Run: ``pytest --forked tests/moe/test_unified_moe_api_fuzz.py``.
Env: FLASHINFER_UMOE_FUZZ_NUM_TESTS (default 80), _SEED (default 0).
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

from flashinfer.fp4_quantization import fp4_quantize
from flashinfer.fused_moe import MoEActivationPack, MoELayer, MoEWeightPack
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
    TrtllmFp4Config,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.quantization import e2m1_and_ufp8sf_scale_to_float
from flashinfer.utils import get_compute_capability

NUM_TESTS = int(os.environ.get("FLASHINFER_UMOE_FUZZ_NUM_TESTS", "80"))
BASE_SEED = int(os.environ.get("FLASHINFER_UMOE_FUZZ_SEED", "0"))

# Per-backend determinism contract, established empirically (CRC across reruns) + confirmed against
# code. A "True" backend MUST reproduce bitwise; flip to False only with evidence (and ideally an
# upstream note), because a deterministic->non-deterministic regression is exactly a bug to catch.
_DETERMINISTIC = {
    "trtllm_fp4_routed": True,  # bitwise-stable across reruns in calibration
    "cute_dsl_nvfp4": False,  # atomic scatter-add finalize -> non-bit-exact by design
}

# Known-bug ledger: (backend_key, predicate(cfg)) -> reason. A matching (backend, config) is run but
# its correctness failure is TOLERATED (xfail) -- this keeps the suite green on a filed-and-tracked
# bug while still EXERCISING it, so the day the bug is fixed the case starts passing and we get a loud
# "unexpectedly passed -> remove this entry" signal. A crash is never tolerated (only wrong answers).
_KNOWN_FAILURES = [
    (
        "trtllm_fp4_routed",
        lambda c: c.expert_offset > 0,
        "trtllm EP local_expert_offset>0 -> all-zero output (offset applied twice); gh #3547",
    ),
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
    make_act_pack: Callable  # (x, selected_experts, final_scales) -> MoEActivationPack
    reference: Callable  # (x, w1, w2, selected_experts, final_scales, I) -> fp32 [T,H] authority
    poison: Callable  # in-place fill a kernel-owned output buffer with garbage + (NaN/Inf if repr.)
    out_dtype: torch.dtype  # output buffer dtype (used to locate it in the inputs list)
    atol_frac: float  # numeric tolerance vs reference = atol_frac * ‖ref‖∞
    rtol: float


def _nvfp4_poison(buf):
    """Fill a bf16 output buffer with large garbage + scattered NaN/±Inf. If a kernel reads or
    scatter-adds into an uninitialized output instead of fully writing it, the poison leaks and
    is caught by no-NaN / numeric. This is the torch->JAX buffer-hygiene guard: torch's caching
    allocator usually hands back clean memory (masking the bug), JAX/XLA donates dirty buffers
    (the GH-6158764 class)."""
    g = torch.randn_like(buf) * 1e4
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
        selected_experts=selected_experts,
        final_scales=final_scales,
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


_DTYPE = {
    QuantVariant.NVFP4: DTypeHandler(
        variant=QuantVariant.NVFP4,
        candidate_configs=(CuteDslConfig, TrtllmFp4Config),
        snap=_snap_to_nvfp4,
        make_act_pack=_nvfp4_act_pack,
        reference=_nvfp4_reference,
        poison=_nvfp4_poison,
        out_dtype=torch.bfloat16,
        atol_frac=0.15,  # calibrated: obs ratio ≤0.077 (fp4 intermediate-requant floor)
        rtol=0.1,
    ),
    # FP8 / MXFP4 / MXINT4 / BF16 add one entry each as their runners are wired upstream.
}


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
_ROUTE = ["uniform", "uniform", "hot1", "imbalanced"]

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

    @property
    def n_local(self):  # experts actually held + computed on this rank
        return self.local_experts or self.num_experts

    @property
    def is_ep(self):
        return self.expert_offset > 0 or self.n_local != self.num_experts

    @property
    def label(self):
        ep = f"L{self.n_local}o{self.expert_offset}_" if self.is_ep else ""
        return (
            f"{self.variant}_{self.route}_e{self.num_experts}_{ep}k{self.top_k}_"
            f"t{self.num_tokens}_h{self.hidden}_i{self.intermediate}_s{self.seed}"
        )


def _gen(seed):
    rng = random.Random(seed)
    # Resample shape until the LOCAL-shard weights fit the budget (modest per-test GPU footprint).
    for _ in range(64):
        ne, h, i = rng.choice(_EXPERTS), rng.choice(_HIDDEN), rng.choice(_INTERMED)
        # ~30%: expert-parallel shard -- split the global experts and pick a shard (offset>0). This
        # is how large MoE actually runs (no rank holds all experts) and exercises the offset path.
        local, offset = ne, 0
        shards = rng.choice([2, 4])
        if rng.random() < 0.3 and ne >= 16 and ne % shards == 0:
            local = ne // shards
            offset = local * rng.randrange(shards)
        if _weight_elems(local, h, i) <= _WEIGHT_ELEM_BUDGET:
            break
    return Cfg(
        num_tokens=rng.choice(_TOKENS),
        hidden=h,
        intermediate=i,
        num_experts=ne,
        top_k=rng.choice(
            [t for t in _TOPK if t <= local]
        ),  # route within the local shard
        variant="nvfp4",  # only wired variant today; expands with _DTYPE
        route=rng.choice(_ROUTE),
        seed=seed,
        local_experts=local,
        expert_offset=offset,
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
]
_CONFIGS = _CURATED + [_gen(BASE_SEED + i) for i in range(NUM_TESTS)]


def _master(cfg, handler):
    """Sparse, exactly-representable bf16 inputs + host routing. Sparsity keeps the gemm reductions
    short so a structural bug isn't averaged away; exact-grid snapping makes input quant lossless.
    Weights cover only this rank's LOCAL shard (E_local); routing selects within the shard's GLOBAL
    id range [offset, offset+E_local) (the EP contract -- non-EP is offset=0, E_local=num_experts)."""
    g = torch.Generator(device="cuda").manual_seed(cfg.seed)
    E_local, H, I, T = cfg.n_local, cfg.hidden, cfg.intermediate, cfg.num_tokens

    def sparse(*shape):
        dense = torch.randn(*shape, device="cuda", generator=g)
        keep = torch.rand(shape, device="cuda", generator=g) >= 0.75  # ~75% zeros
        return handler.snap(dense * keep)

    x, w1, w2 = sparse(T, H), sparse(E_local, 2 * I, H), sparse(E_local, H, I)

    logits = torch.randn(T, E_local, device="cuda", generator=g)  # over the local shard
    if cfg.route == "hot1":  # pile every token onto one expert
        logits[:, 0] += 50.0
    elif cfg.route == "imbalanced":  # rank-skew -> some experts get zero tokens
        logits += torch.linspace(8.0, -8.0, E_local, device="cuda")
    weights = F.softmax(logits, dim=1, dtype=torch.float32)
    weights, local_sel = torch.topk(weights, cfg.top_k, dim=-1)
    final_scales = (weights / weights.sum(dim=-1, keepdim=True)).float()
    selected_experts = (local_sel + cfg.expert_offset).to(
        torch.int32
    )  # local -> global ids
    return x, w1, w2, selected_experts, final_scales


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


@pytest.mark.parametrize("cfg", _CONFIGS, ids=[c.label for c in _CONFIGS])
def test_unified_moe_api_fuzz(cfg):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    sm = get_compute_capability(torch.device("cuda:0"))
    sm = sm[0] * 10 + sm[1]

    handler = _DTYPE[QuantVariant.NVFP4]
    dev = torch.device("cuda")
    # Backend *config classes* whose runner is registered in the live MoELayer registry AND valid
    # on this arch. A newly-wired backend lands here automatically.
    wired_backends = [
        BackendCfg
        for BackendCfg in handler.candidate_configs
        if BackendCfg in _BACKEND_RUNNERS and BackendCfg.supported(sm)
    ]
    if not wired_backends:
        pytest.skip(f"no wired backend for {cfg.variant} on SM{sm}")

    x, w1, w2, selected_experts, final_scales = _master(cfg, handler)
    ref = handler.reference(
        x, w1, w2, selected_experts, final_scales, cfg.intermediate, cfg.expert_offset
    )
    atol = handler.atol_frac * ref.abs().max().item() + 1e-3
    rtol = handler.rtol

    # One activation pack + one weight pack with each backend's native view, all built from the
    # SAME bf16 inputs (this rank's LOCAL shard) via the API's uniform prepare_weights.
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
        routing=RoutingConfig(num_experts=cfg.num_experts, top_k=cfg.top_k),
        quant=QuantConfig(variant=QuantVariant.NVFP4),
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
        if poison:
            # The output buffer is a kernel-owned `new_empty` tensor inside the inputs list
            # (cute_dsl idx 11, trtllm the `output=`); locate it by dtype+shape and poison it.
            bufs = [
                t
                for t in inputs
                if torch.is_tensor(t)
                and t.dtype == handler.out_dtype
                and tuple(t.shape) == out_shape
            ]
            assert bufs, "could not locate the output buffer in pack_inputs to poison"
            for b in bufs:
                handler.poison(b)
        out = runner.forward(inputs, tactic=-1)
        out = (out[0] if isinstance(out, (list, tuple)) else out).float()
        torch.cuda.synchronize()
        return out

    def assert_correct(out, tag):
        # no NaN/Inf where the reference is finite.
        n_bad = int(((~torch.isfinite(out)) & torch.isfinite(ref)).sum().item())
        assert n_bad == 0, f"{tag}: {n_bad} non-finite outputs vs finite reference"
        # numeric vs the canonical quant-aware reference (the authority), magnitude-scaled.
        abs_diff = (out - ref).abs()
        over_tol = abs_diff > (atol + rtol * ref.abs())
        if over_tol.any():
            pytest.fail(
                f"{tag}: {int(over_tol.sum())}/{out.numel()} elems exceed "
                f"(rtol={rtol} atol={atol:.3g}); max|diff|={abs_diff.max().item():.4g}, "
                f"‖ref‖∞={ref.abs().max().item():.4g}"
            )

    def check_backend(runner, out, tag):
        # (1)+(2) no-NaN + numeric vs the authoritative reference, on a clean run.
        assert_correct(out, tag)
        # (3) determinism per the backend's contract: deterministic backends must reproduce
        # bitwise; non-deterministic ones (atomic-scatter finalize) are exempt.
        if _DETERMINISTIC.get(runner.backend_key, False):
            if not torch.equal(out, run(runner)):
                drift = (out - run(runner)).abs().max().item()
                pytest.fail(
                    f"{tag}: declared DETERMINISTIC but not bitwise-reproducible "
                    f"(max abs diff {drift:.3e})"
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

    # (6) device-state probe: a context-corrupting IMA in any backend above would surface here as a
    # failed alloc/launch or a non-finite probe, turning a silent corruption into a clean failure.
    probe = torch.randn(2048, device="cuda") * 2.0
    torch.cuda.synchronize()
    assert torch.isfinite(probe).all(), (
        f"{cfg.label}: CUDA context corrupted after MoE run"
    )
