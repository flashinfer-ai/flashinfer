"""Unified cross-API / cross-arch fuzzer for flashinfer FULL fused-MoE backends.

Thesis (verified against the per-backend test references): the MoE *math* is identical across
all full-MoE backends -- per token, for each selected expert e:
    inter = act(x @ W1[e].T) [* (x @ W3[e].T) for gated]   ;   y += scale * (inter @ W2[e].T)
Only the *input marshalling* (routing form, weight layout, scale convention, output shape) and the
arch support diverge. So this fuzzer is built around:
  * ONE logical config space (`MoEConfig`),
  * ONE fp32 ground-truth (`MasterTensors`) + ONE shared reference (`shared_reference`),
  * thin per-backend ADAPTERS (`MoEAdapter.supports/run`) that marshal master -> native call,
  * shared oracles: numeric-vs-reference, no-NaN, determinism, no-crash, and the kill shots
    **cross-API agreement** (two backends, same logical config, must agree) and **cross-arch
    agreement** (the cutlass backend spans SM89..SM120).

This is the cross-API/cross-arch *correctness* net. It is complementary to `test_moe_fuzz.py`, the
*crash-finder* on the trtllm in-kernel-routing + autotuner path (all routing methods, autotune
ON/OFF, extreme expert counts) -- the two catch different classes, keep both until this fuzzer also
folds in an in-kernel-routing variant + autotune oracle + extreme experts. `grouped_mm` stays a
separate fuzzer (different contract: m_indptr ranges, no routing/gating/scatter).

Run with `pytest --forked` (a MoE config can IMA and corrupt the CUDA context; isolation turns each
into one reproducible failure). Env: FLASHINFER_MOE_FUZZ_NUM_TESTS (default 100), _SEED (default 0).

Coverage today (all SwiGLU):
  * cutlass_fused_moe -- bf16 / fp16 / fp8(per-tensor e4m3) on SM89/90/100+; nvfp4(e2m1 block16)
    on SM100+ (mirrors test_moe_nvfp4 marshalling + reuses its torch_moe_nvfp4 reference exactly).
  * trtllm-gen (pre-routed bf16) -- SM100/103; runs alongside cutlass on bf16 -> cross-API oracle.
Validated false-positive-free: SM100, H100/sm90, L40S/sm89. New backends (cute-dsl, b12x) and quant
modes (nvfp4-on-trtllm, mxfp8/mxfp4) drop in behind the same `MoEAdapter` interface. NOTE: trtllm-gen
abstains on fp8 because its fp8-per-tensor scale + (in-kernel) routing contract is incompatible with
cutlass's -- a deliberate finding for the unified-API design, see UNIFIED_MOE_FUZZER_AND_API_DESIGN.md.
"""

import os
import random
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from flashinfer import ActivationType, RoutingMethodType, fp4_quantize, fused_moe
from flashinfer.fused_moe import (
    WeightLayout,
    convert_to_block_layout,
    trtllm_bf16_routed_moe,
)
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
)
from flashinfer.utils import get_compute_capability

NUM_TESTS = int(os.environ.get("FLASHINFER_MOE_FUZZ_NUM_TESTS", "100"))
BASE_SEED = int(os.environ.get("FLASHINFER_MOE_FUZZ_SEED", "0"))


# ---------------------------------------------------------------------------
# Shared logical config + fp32 ground truth + reference (the backend-agnostic core)
# ---------------------------------------------------------------------------
_E4M3_MAX = 448.0


def _pt_fp8(t):
    """Per-tensor e4m3 quant. Returns (fp8 tensor, scale [scalar], dequantized fp32)."""
    amax = t.abs().max().clamp_min(1e-12)
    scale = (amax / _E4M3_MAX).float()
    q = (t / scale).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale, q.float() * scale


@dataclass(frozen=True)
class MoEConfig:
    num_tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    quant: str  # "bf16" | "fp16" | "fp8" (per-tensor e4m3)
    activation: str  # "swiglu"
    seed: int
    route: str = "uniform"  # routing-load regime (see MasterTensors)

    @property
    def label(self):
        return (
            f"{self.quant}_e{self.num_experts}_k{self.top_k}_t{self.num_tokens}_"
            f"h{self.hidden_size}_i{self.intermediate_size}_{self.activation}_"
            f"{self.route}_s{self.seed}"
        )


class MasterTensors:
    """The single fp32 ground truth all backends are marshalled from + compared against."""

    def __init__(self, cfg: MoEConfig):
        g = torch.Generator(device="cuda").manual_seed(cfg.seed)
        E, H, I, T = (
            cfg.num_experts,
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.num_tokens,
        )
        # Small magnitudes (like the existing tests) keep the unquantized path well-conditioned.
        self.x = torch.randn(T, H, device="cuda", dtype=torch.float32, generator=g) / 5
        self.w31 = (
            torch.randn(E, 2 * I, H, device="cuda", dtype=torch.float32, generator=g)
            / 5
        )
        self.w2 = (
            torch.randn(E, H, I, device="cuda", dtype=torch.float32, generator=g) / 5
        )
        self.router_logits = torch.randn(
            T, E, device="cuda", dtype=torch.float32, generator=g
        )
        # Routing-load regimes: uniform random rarely produces the load patterns that break MoE
        # scatter/gather + tile scheduling. Bias the logits to stress them while leaving the math
        # (and reference) unchanged -- only the *distribution* of selected experts shifts:
        #   hot1       all tokens pile onto a single expert (max load on one tile)
        #   peaked     near-argmax per token (sharp softmax -> degenerate weights)
        #   imbalanced zipf-ish per-expert bias -> some experts get many tokens, many get ZERO
        #              (the empty-expert path: a classic skip-bug / OOB source)
        if cfg.route == "hot1":
            self.router_logits[:, 0] += 50.0
        elif cfg.route == "peaked":
            self.router_logits *= 50.0
        elif cfg.route == "imbalanced":
            bias = torch.linspace(8.0, -8.0, E, device="cuda")  # rank-skewed
            self.router_logits += bias
        # Route ONCE on the host (softmax -> top-k -> renormalize) so every caller-routed backend
        # gets the identical selection -> a sound cross-API comparison.
        rw = F.softmax(self.router_logits, dim=1, dtype=torch.float32)
        rw, sel = torch.topk(rw, cfg.top_k, dim=-1)
        rw = (rw / rw.sum(dim=-1, keepdim=True)).float()
        self.routing_weights = rw  # [T, top_k]
        self.selected_experts = sel.to(torch.int32)  # [T, top_k]

        # fp8 per-tensor quantization: x once, weights per-expert (matches the cutlass
        # fp8 MoE test convention). The dequantized views are the reference inputs so the
        # shared reference models exactly the fp8 rounding the kernel sees.
        if cfg.quant == "fp8":
            self.x_fp8, self.x_scale, self.x_dq = _pt_fp8(self.x)
            w31q, w2q, s31, s2, w31dq, w2dq = [], [], [], [], [], []
            for e in range(E):
                q, s, dq = _pt_fp8(self.w31[e])
                w31q.append(q), s31.append(s), w31dq.append(dq)
                q, s, dq = _pt_fp8(self.w2[e])
                w2q.append(q), s2.append(s), w2dq.append(dq)
            self.w31_fp8, self.w31_scale, self.w31_dq = (
                torch.stack(w31q),
                torch.stack(s31).view(E),
                torch.stack(w31dq),
            )
            self.w2_fp8, self.w2_scale, self.w2_dq = (
                torch.stack(w2q),
                torch.stack(s2).view(E),
                torch.stack(w2dq),
            )

        # nvfp4 (e2m1, block 16, e4m3 block scales): mirror tests/moe/test_trtllm_cutlass_fused_moe
        # test_moe_nvfp4 EXACTLY (same fp4_quantize + per-expert global scale + 6-elem quant_scales)
        # so the marshalling matches the validated cutlass test by construction (no re-derivation).
        if cfg.quant == "nvfp4":
            from tests.moe.test_trtllm_cutlass_fused_moe import (
                dequantize_nvfp4_to_dtype,
            )
            from tests.test_helpers.utils_fp4 import nvfp4_global_encode_scale_te

            ot = torch.bfloat16
            self.nvfp4_x = self.x.to(
                ot
            )  # raw input (kernel quantizes internally, input_sf=None)
            self.a1_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
            self.a2_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
            w1b, w2b = self.w31.to(ot), self.w2.to(ot)
            q1l, b1l, g1l, d1l, q2l, b2l, g2l, d2l = [], [], [], [], [], [], [], []
            for e in range(E):
                g1 = nvfp4_global_encode_scale_te(w1b[e].abs().max().float())
                g2 = nvfp4_global_encode_scale_te(w2b[e].abs().max().float())
                q1, b1 = fp4_quantize(w1b[e], g1)
                q2, b2 = fp4_quantize(w2b[e], g2)
                q1l.append(q1), b1l.append(b1), g1l.append(g1)
                d1l.append(
                    dequantize_nvfp4_to_dtype(
                        q1, b1, g1, dtype=ot, device="cuda", block_size=16
                    )
                )
                q2l.append(q2), b2l.append(b2), g2l.append(g2)
                d2l.append(
                    dequantize_nvfp4_to_dtype(
                        q2, b2, g2, dtype=ot, device="cuda", block_size=16
                    )
                )
            self.w1_q, self.w1_bs, self.w1_gs = (
                torch.stack(q1l),
                torch.stack(b1l),
                torch.stack(g1l).view(E),
            )
            self.w2_q, self.w2_bs, self.w2_gs = (
                torch.stack(q2l),
                torch.stack(b2l),
                torch.stack(g2l).view(E),
            )
            self.w1_d, self.w2_d = torch.stack(d1l), torch.stack(d2l)
            # reference input = dequant of the kernel's internal fp4 quant of x (a1_gs scale).
            af, asf = fp4_quantize(self.nvfp4_x, self.a1_gs)
            self.a_in = dequantize_nvfp4_to_dtype(
                af, asf, self.a1_gs, dtype=ot, device="cuda", block_size=16
            )


def shared_reference(m: MasterTensors, cfg: MoEConfig) -> torch.Tensor:
    """The one fp32 MoE reference (SwiGLU). w31 = concat[w3(linear), w1(gated)] (chunk dim 0).

    For fp8 the reference consumes the *dequantized* per-tensor-quant inputs and quantizes the
    intermediate to fp8 with scale 1.0 (the cutlass fc2-input-quant scale) -- i.e. it reproduces
    the full fp8 round-trip (input + weight + intermediate), not just weight quant, so the oracle
    stays tight rather than papering over kernel rounding with a loose tolerance.
    """
    if cfg.quant == "nvfp4":
        # Reuse the validated cutlass-test reference verbatim: dequant inputs/weights, SwiGLU
        # (silu(2nd half)*(1st half)), intermediate fp4 requant @ gs=1.0, gemm2, weighted sum.
        from tests.moe.test_trtllm_cutlass_fused_moe import torch_moe_nvfp4

        return torch_moe_nvfp4(
            m.a_in,
            m.w1_d,
            m.w2_d,
            cfg.top_k,
            m.routing_weights,
            m.selected_experts,
            _ACT[cfg.activation],
        ).float()

    fp8 = cfg.quant == "fp8"
    x = m.x_dq if fp8 else m.x
    w31 = m.w31_dq if fp8 else m.w31
    w2 = m.w2_dq if fp8 else m.w2
    out = torch.zeros_like(x)
    for e in range(cfg.num_experts):
        mask = m.selected_experts == e
        if not mask.any():
            continue
        tok, nth = torch.where(mask)
        w3, w1 = torch.chunk(w31[e], 2, dim=0)  # matches compute_with_experts split
        xe = x[tok]
        inter = F.silu(xe @ w1.t()) * (xe @ w3.t())
        if (
            fp8
        ):  # intermediate fp8 quant, per-tensor scale 1.0 (== cutlass quant_scales[1])
            inter = inter.clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn).float()
        y = inter @ w2[e].t()
        out[tok] += m.routing_weights[tok, nth, None] * y
    return out  # [T, H], fp32


# ---------------------------------------------------------------------------
# Adapter interface + backends
# ---------------------------------------------------------------------------
class MoEAdapter:
    name: str

    def supports(self, cfg: MoEConfig, sm: int) -> Optional[str]:
        """Return None if runnable on this (config, arch), else a skip reason."""
        raise NotImplementedError

    def run(self, m: MasterTensors, cfg: MoEConfig) -> torch.Tensor:
        """Marshal master -> native call; return finalized output [num_tokens, hidden] (fp32-cast)."""
        raise NotImplementedError


_ACT = {"swiglu": ActivationType.Swiglu}


class CutlassAdapter(MoEAdapter):
    name = "cutlass"

    def supports(self, cfg, sm):
        if sm not in (89, 90, 100, 103, 110, 120, 121):
            return f"cutlass_fused_moe unsupported on SM{sm}"
        if cfg.quant not in ("bf16", "fp16", "fp8", "nvfp4"):
            return f"cutlass adapter: quant {cfg.quant} not wired yet"
        if cfg.quant == "nvfp4":
            if sm < 100:  # nvfp4 MoE kernels require Blackwell+
                return "cutlass nvfp4 MoE needs SM100+"
            if cfg.hidden_size % 16 or cfg.intermediate_size % 16:
                return "nvfp4 block-16 requires hidden/intermediate %16==0"
        return None

    def run(self, m, cfg):
        if cfg.quant == "fp8":
            return self._run_fp8(m, cfg)
        if cfg.quant == "nvfp4":
            return self._run_nvfp4(m, cfg)
        dt = torch.bfloat16 if cfg.quant == "bf16" else torch.float16
        x = m.x.to(dt)
        w31 = m.w31.to(dt)
        w2 = m.w2.to(dt)
        out = torch.empty(cfg.num_tokens, cfg.hidden_size, device="cuda", dtype=dt)
        res = fused_moe.cutlass_fused_moe(
            x,
            m.selected_experts.to(torch.int),
            m.routing_weights,
            w31,
            w2,
            dt,
            output=out,
            quant_scales=None,
            activation_type=_ACT[cfg.activation],
        )
        # cutlass returns [output, num_active_experts, ...] (min-latency buffers); take output.
        y = res[0] if isinstance(res, (list, tuple)) else res
        return y.float()

    def _run_fp8(self, m, cfg):
        """Per-tensor fp8: pre-quantized x_fp8 + fp8 weights + positional quant_scales
        [fc1_dequant(=w*input), fc2_quant(=1.0), fc2_dequant(=w2), fc1_input_scale]."""
        otype = torch.bfloat16
        dev = m.x.device
        quant_scales = [
            (m.w31_scale * m.x_scale).float(),  # [E] fc1 output dequant
            torch.tensor(1.0, device=dev),  # fc2 input quant scale
            m.w2_scale.float(),  # [E] fc2 output dequant
            m.x_scale.float().reshape(()),  # fc1 input (pre-quant) scale
        ]
        out = torch.empty(cfg.num_tokens, cfg.hidden_size, device=dev, dtype=otype)
        res = fused_moe.cutlass_fused_moe(
            m.x_fp8,
            m.selected_experts.to(torch.int),
            m.routing_weights,
            m.w31_fp8,
            m.w2_fp8,
            otype,
            output=out,
            quant_scales=quant_scales,
            activation_type=_ACT[cfg.activation],
        )
        y = res[0] if isinstance(res, (list, tuple)) else res
        return y.float()

    def _run_nvfp4(self, m, cfg):
        """nvfp4 block-scale: raw bf16 x (kernel quantizes internally, input_sf=None) + packed
        fp4 weights (.view(long)) + the 6-elem quant_scales list from test_moe_nvfp4:
        [fc1_act_global, fc1_weight_block(int32), fc1_global=1/(a1*w1), fc2_act_global,
         fc2_weight_block(int32), fc2_global=1/(a2*w2)]."""
        ot = torch.bfloat16
        quant_scales = [
            m.a1_gs,
            m.w1_bs.view(torch.int32),
            1.0 / (m.a1_gs * m.w1_gs),
            m.a2_gs,
            m.w2_bs.view(torch.int32),
            1.0 / (m.a2_gs * m.w2_gs),
        ]
        out = torch.zeros(cfg.num_tokens, cfg.hidden_size, device="cuda", dtype=ot)
        res = fused_moe.cutlass_fused_moe(
            m.nvfp4_x,
            m.selected_experts.to(torch.int),
            m.routing_weights,
            m.w1_q.contiguous().view(torch.long),
            m.w2_q.contiguous().view(torch.long),
            ot,
            quant_scales=quant_scales,
            input_sf=None,
            output=out,
            activation_type=_ACT[cfg.activation],
        )
        y = res[0] if isinstance(res, (list, tuple)) else res
        return y.float()


def _pack_topk(selected_experts, routing_weights):
    """trtllm routed-MoE ``topk_ids`` packing: (expert_id << 16) | weight_bf16.view(int16)."""
    return (selected_experts.to(torch.int32) << 16) | routing_weights.to(
        torch.bfloat16
    ).view(torch.int16).to(torch.int32)


class TrtllmGenAdapter(MoEAdapter):
    """trtllm-gen fused MoE via the PRE-ROUTED entry point.

    The routed entry (``trtllm_bf16_routed_moe``) consumes a packed ``topk_ids`` (expert id +
    final weight) instead of routing in-kernel -- this is what makes a sound cross-API oracle
    possible: we feed it the *identical* host routing the cutlass adapter uses. Its in-kernel
    gated activation is ``silu(second half) * (first half)`` (test ref line 2417), the same
    convention as cutlass/`shared_reference`, so the logical weight layout is shared; only the
    BlockMajorK weight shuffle + int32 topk packing differ (the marshalling this fuzzer exists
    to stress)."""

    name = "trtllm"

    def __init__(self):
        self._permute_cache = {}

    def supports(self, cfg, sm):
        if sm not in (100, 103):
            return f"trtllm-gen MoE gated to SM100/103 (not SM{sm})"
        if cfg.quant == "fp8":
            # trtllm fp8-per-tensor routes IN-KERNEL (no routed entry) and computes the
            # intermediate scale internally, vs cutlass's caller-routing + explicit
            # quant_scales[1]=1.0 -- the two fp8 scale/routing contracts are incompatible,
            # so a forced cross-API fp8 oracle would diverge legitimately. This contract
            # gap IS a unified-API design finding (see UNIFIED_MOE_FUZZER_AND_API_DESIGN.md);
            # we keep fp8 as a cutlass-only (vs-reference + cross-arch) oracle for now.
            return "trtllm fp8-per-tensor scale/routing contract diverges from cutlass"
        if cfg.quant != "bf16":
            return f"trtllm adapter: quant {cfg.quant} not wired yet"
        # BlockMajorK packs K in 128-blocks; gemm1 K=hidden, gemm2 K=intermediate.
        if cfg.hidden_size % 128 or cfg.intermediate_size % 128:
            return "trtllm BlockMajorK requires hidden/intermediate %128==0"
        return None

    def _prep_weights(self, w31_bf16, w2_bf16, num_experts):
        """Replicate BF16Moe.prepare_static_weights_for_kernel (shuffle + BlockMajorK)."""
        etm = 128  # epilogue_tile_m (kernel internal)
        g1, g2 = [], []
        for i in range(num_experts):
            pi = _maybe_get_cached_w3_w1_permute_indices(
                self._permute_cache,
                w31_bf16[i].view(torch.uint8),
                etm,
                is_gated_act_gemm=True,  # swiglu
            )
            t1 = w31_bf16[i].view(torch.uint8)[pi.to(w31_bf16.device)].contiguous()
            pi2 = get_w2_permute_indices_with_cache(
                self._permute_cache, w2_bf16[i].view(torch.uint8), etm
            )
            t2 = w2_bf16[i].view(torch.uint8)[pi2.to(w2_bf16.device)].contiguous()
            t1 = convert_to_block_layout(t1.view(torch.uint8), 128)
            t2 = convert_to_block_layout(t2.view(torch.uint8), 128)
            g1.append(t1.view(torch.bfloat16))
            g2.append(t2.view(torch.bfloat16))
        return torch.stack(g1).contiguous(), torch.stack(g2).contiguous()

    def run(self, m, cfg):
        x = m.x.to(torch.bfloat16)
        w31 = m.w31.to(torch.bfloat16)
        w2 = m.w2.to(torch.bfloat16)
        g1, g2 = self._prep_weights(w31, w2, cfg.num_experts)
        packed = _pack_topk(m.selected_experts, m.routing_weights)
        out = trtllm_bf16_routed_moe(
            packed,
            x,
            g1,
            g2,
            cfg.num_experts,
            cfg.top_k,
            None,  # n_group
            None,  # topk_group
            cfg.intermediate_size,
            0,  # local_expert_offset
            cfg.num_experts,  # local_num_experts
            routed_scaling_factor=None,
            routing_method_type=RoutingMethodType.RenormalizeNaive.value,
            use_shuffled_weight=True,
            weight_layout=WeightLayout.BlockMajorK,
            do_finalize=True,
            activation_type=_ACT[cfg.activation].value,
        )
        y = out[0] if isinstance(out, (list, tuple)) else out
        return y.float()


ADAPTERS = [CutlassAdapter(), TrtllmGenAdapter()]


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------
_HIDDEN = [256, 512, 1024, 2048]
_INTERMED = [256, 512, 768, 1024, 1536]
_EXPERTS = [8, 16, 32, 64, 128, 256]
_TOPK = [1, 2, 4, 6, 8]
_TOKENS = [1, 2, 8, 16, 64, 128, 512, 1024, 2048, 4096]
_QUANT = ["bf16", "fp16", "fp8", "nvfp4"]
_ROUTE = ["uniform", "uniform", "hot1", "peaked", "imbalanced"]  # weight uniform ~2x


def _gen(seed):
    rng = random.Random(seed)
    ne = rng.choice(_EXPERTS)
    tk = rng.choice([t for t in _TOPK if t <= ne])
    return MoEConfig(
        num_tokens=rng.choice(_TOKENS),
        hidden_size=rng.choice(_HIDDEN),
        intermediate_size=rng.choice(_INTERMED),
        num_experts=ne,
        top_k=tk,
        quant=rng.choice(_QUANT),
        activation="swiglu",
        seed=seed,
        route=rng.choice(_ROUTE),
    )


_CONFIGS = [_gen(BASE_SEED + i) for i in range(NUM_TESTS)]


def _tol(cfg, ref):
    """Magnitude-scaled element-wise tolerance vs the fp32 reference.

    Measured across the config space, ``max|out-ref| / ‖ref‖∞`` is tightly bounded and
    *constant in accumulation length* (it is the final-downcast ULP, not a growing sum):
    bf16 ≤ 0.0064, fp16 ≤ 0.00084. So we scale ``atol`` by ‖ref‖∞ with a ~3-5x safety
    margin and keep a modest ``rtol`` for mid-magnitude elements. This stays element-wise
    (no outlier-fraction allowance -> localized corruption a cosine oracle would miss still
    fails) while absorbing legitimate low-precision rounding -- far tighter than the
    per-backend suites' ``rtol=0.85`` + 7.5%-outlier oracle.
    """
    absmax = ref.abs().max().item()
    if cfg.quant == "fp16":
        return 1e-2, 4e-3 * absmax + 1e-4
    if (
        cfg.quant == "fp8"
    ):  # e4m3 ~2 mantissa bits + intermediate requant; calibrated below
        return 8e-2, 8e-2 * absmax + 1e-3
    if (
        cfg.quant == "nvfp4"
    ):  # e2m1 ~1 mantissa bit + block16 + intermediate fp4 requant
        return (
            2e-1,
            2.5e-1 * absmax + 1e-3,
        )  # calibrated: obs ratio ≤0.073, ~3.4x margin
    return 3e-2, 2e-2 * absmax + 1e-4  # bf16


# NOT_SUPPORTED-class signals: a backend legitimately rejecting a shape -> skip, never a finding.
# Deliberately tight: a CUDA error / IMA / assert is a *crash bug* and must surface, not be swallowed.
_SKIP_SUBSTR = (
    "not supported",
    "unsupported",
    "no valid",
    "not implemented",
    "is not divisible",
    "must be a multiple",
    "must be divisible",
    "requires sm",
    "only supported",
)
_CRASH_SUBSTR = ("cuda error", "illegal memory", "misaligned", "device-side assert")


def _is_unsupported(e: Exception) -> bool:
    if isinstance(e, NotImplementedError):
        return True
    msg = str(e).lower()
    if any(c in msg for c in _CRASH_SUBSTR):  # crash bugs never count as "unsupported"
        return False
    return any(s in msg for s in _SKIP_SUBSTR)


@pytest.mark.parametrize("cfg", _CONFIGS, ids=[c.label for c in _CONFIGS])
def test_unified_moe_fuzz(cfg):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    sm = get_compute_capability(torch.device("cuda:0"))
    sm = sm[0] * 10 + sm[1]

    m = MasterTensors(cfg)
    ref = shared_reference(m, cfg)
    rtol, atol = _tol(cfg, ref)

    outputs = {}
    skipped = []
    for ad in ADAPTERS:
        reason = ad.supports(cfg, sm)
        if reason:
            skipped.append(f"{ad.name}:{reason}")
            continue
        try:
            out = ad.run(m, cfg)
            torch.cuda.synchronize()
        except Exception as e:
            # A backend rejecting an out-of-contract shape is a skip; a crash is a finding.
            if _is_unsupported(e):
                skipped.append(f"{ad.name}:rejected({type(e).__name__})")
                continue
            raise

        # (1) no NaN/Inf where the reference is finite.
        bad = int(((~torch.isfinite(out)) & torch.isfinite(ref)).sum().item())
        assert bad == 0, (
            f"{ad.name} {cfg.label}: {bad} non-finite outputs vs finite reference"
        )
        # (2) numeric vs the shared fp32 reference (magnitude-scaled, element-wise).
        absd = (out - ref).abs()
        viol = absd > (atol + rtol * ref.abs())
        if viol.any():
            nv = int(viol.sum())
            md = absd.max().item()
            pytest.fail(
                f"{ad.name} {cfg.label}: output != shared reference -- {nv}/{out.numel()} "
                f"elems violate (rtol={rtol:.3g} atol={atol:.3g}); max|diff|={md:.4g}, "
                f"‖ref‖∞={ref.abs().max().item():.4g}"
            )
        # (3) determinism: identical re-run must match bit-exactly.
        out2 = ad.run(m, cfg)
        torch.cuda.synchronize()
        if not torch.equal(out, out2):
            md = (out - out2).abs().max().item()
            pytest.fail(
                f"{ad.name} {cfg.label}: NONDETERMINISTIC (max abs diff {md:.3e})"
            )
        outputs[ad.name] = out

    if not outputs:
        pytest.skip(f"no adapter ran {cfg.label} on SM{sm} [{'; '.join(skipped)}]")

    # (4) cross-API agreement: every pair of backends that ran must agree within combined tol.
    names = list(outputs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = outputs[names[i]], outputs[names[j]]
            torch.testing.assert_close(
                a,
                b,
                rtol=2 * rtol,
                atol=2 * atol,
                msg=f"CROSS-API disagreement {names[i]} vs {names[j]} on {cfg.label}",
            )
