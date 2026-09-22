"""Independent, memory-bounded references for native MXFP4 SiTU MoE.

This is evaluation code, never part of a timed/captured forward. Canonical
weights are packed E2M1 in [up, gate] row order with linear UE8M0 scales.
The primary FP64 reference uses those exact quantized operands and omits
intermediate quantization. Named auxiliary references isolate MXFP8 rounding.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MXFP4SiTUCase:
    x: torch.Tensor
    x_scale: torch.Tensor
    w1: torch.Tensor
    w1_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    beta: torch.Tensor
    linear_beta: Optional[torch.Tensor]
    num_experts: int
    local_expert_offset: int

    @property
    def local_num_experts(self):
        return self.w1.shape[0]

    @property
    def hidden_size(self):
        return self.x.shape[1]

    @property
    def intermediate_size(self):
        return self.w2.shape[2] * 2


def decode_ue8m0(scales: torch.Tensor, dtype=torch.float64) -> torch.Tensor:
    """Decode raw bytes: codes 0..254 mean 2**(code-127), 255 is NaN."""
    codes = scales.view(torch.uint8).to(torch.int32)
    values = torch.ldexp(torch.ones_like(codes, dtype=dtype), codes - 127)
    return torch.where(codes == 255, torch.full_like(values, float("nan")), values)


def decode_mxfp4(packed, scales, *, device=None, dtype=torch.float64):
    """Decode canonical row-major nibbles; low nibble is the even K index."""
    packed = packed.to(device=device, dtype=torch.uint8)
    scales = scales.to(device=packed.device).view(torch.uint8)
    table = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=packed.device,
        dtype=dtype,
    )
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    values = table[nibbles.long()]
    blocks = values.reshape(*values.shape[:-1], -1, 32)
    return (blocks * decode_ue8m0(scales, dtype).unsqueeze(-1)).flatten(-2)


def decode_mxfp8(values, scales, *, dtype=torch.float64):
    blocks = values.to(dtype).reshape(*values.shape[:-1], -1, 32)
    return (blocks * decode_ue8m0(scales, dtype).unsqueeze(-1)).flatten(-2)


def quantize_mxfp8_reference(values: torch.Tensor):
    """FP32 -> MXFP8, group 32, upward UE8M0 scale and nearest E4M3.

    Scaling uses FP32 multiplication by 1/448, like the CuTe epilogue.
    Integer exponent extraction avoids log2 boundary errors. This function
    returns linear scale bytes; it does not use either MoE implementation.
    """
    shape = values.shape
    blocks = values.float().reshape(*shape[:-1], -1, 32)
    scale = blocks.abs().amax(dim=-1) * (1.0 / 448.0)
    bits = scale.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    code = (exponent + bump.to(torch.int32)).clamp(0, 254)
    code = torch.where(scale <= 0, torch.zeros_like(code), code).to(torch.uint8)
    # The finite normal range has exact binary-power reciprocals. Handle zero
    # blocks explicitly, without 0*inf for the minimum representable scale.
    inverse = torch.ldexp(torch.ones_like(scale, dtype=torch.float64), 127 - code.int())
    scaled = blocks.double() * inverse.unsqueeze(-1)
    scaled = torch.where((scale == 0).unsqueeze(-1), 0.0, scaled)
    quantized = scaled.clamp(-448.0, 448.0).float().to(torch.float8_e4m3fn)
    return quantized.reshape(shape), code


def pack_topk(topk_ids, topk_weights):
    """Use exactly the BF16 router weights represented by TRT's packed ABI."""
    weights = topk_weights.to(torch.bfloat16).contiguous()
    return (topk_ids.to(torch.int32) << 16) | (weights.view(torch.int16).int() & 0xFFFF)


def make_routing(
    tokens,
    num_experts,
    top_k,
    local_num_experts,
    local_expert_offset=0,
    distribution="balanced",
    *,
    device="cuda",
    seed=17,
):
    """Unique global expert IDs per token with explicit controlled imbalance.

    balanced cycles globally; empty concentrates on a small local subset and
    remote experts; hot assigns one local expert to every token; all_remote
    never selects a local expert; remote_dominated keeps ``top_k // 10`` local
    slots per token (at least 90% of routes leave the local interval) and
    cycles those slots through the local experts. These are synthetic probes,
    not estimates of a production routing distribution.
    """
    if not (0 <= local_expert_offset < num_experts):
        raise ValueError("invalid local expert offset")
    if (
        local_expert_offset + local_num_experts > num_experts
        or not 1 <= top_k <= num_experts
    ):
        raise ValueError("invalid expert geometry")
    t = torch.arange(tokens, dtype=torch.int64)
    k = torch.arange(top_k, dtype=torch.int64)
    if distribution == "balanced":
        # Spread one token across the expert range (and hence EP ranks), then
        # rotate the selected IDs across tokens. Kimi E=896, K=16 gives two
        # assignments per EP8 rank even at T=1.
        ids = (
            t[:, None] + k * (num_experts // top_k) + local_expert_offset
        ) % num_experts
    else:
        local = list(
            range(local_expert_offset, local_expert_offset + local_num_experts)
        )
        remote = [e for e in range(num_experts) if e not in local]
        if distribution == "empty":
            # Use the smallest local subset that can still supply top_k
            # distinct IDs when too few remote experts exist.
            active = local[: max(1, top_k - len(remote))] + remote
            pool = torch.tensor(active, dtype=torch.int64)
            ids = pool[(t[:, None] * top_k + k) % len(active)]
        elif distribution == "hot":
            pool = torch.tensor(
                [e for e in range(num_experts) if e != local[0]], dtype=torch.int64
            )
            ids = torch.empty((tokens, top_k), dtype=torch.int64)
            ids[:, 0] = local[0]
            if top_k > 1:
                ids[:, 1:] = pool[
                    (t[:, None] * (top_k - 1) + torch.arange(top_k - 1)) % len(pool)
                ]
        elif distribution == "all_remote":
            if len(remote) < top_k:
                raise ValueError("not enough remote experts for all_remote")
            pool = torch.tensor(remote, dtype=torch.int64)
            ids = pool[(t[:, None] * top_k + k) % len(pool)]
        elif distribution == "remote_dominated":
            local_slots = top_k // 10
            remote_slots = top_k - local_slots
            if len(remote) < remote_slots:
                raise ValueError("not enough remote experts for remote_dominated")
            pool = torch.tensor(remote, dtype=torch.int64)
            ids = torch.empty((tokens, top_k), dtype=torch.int64)
            ids[:, :remote_slots] = pool[
                (t[:, None] * remote_slots + torch.arange(remote_slots)) % len(pool)
            ]
            if local_slots:
                local_pool = torch.tensor(local, dtype=torch.int64)
                ids[:, remote_slots:] = local_pool[
                    (t[:, None] * local_slots + torch.arange(local_slots))
                    % len(local_pool)
                ]
        else:
            raise ValueError(f"unknown routing distribution: {distribution}")
    generator = torch.Generator().manual_seed(seed)
    weights = torch.rand((tokens, top_k), generator=generator) + 0.125
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
    return ids.to(device=device, dtype=torch.int32), weights.to(device)


def routing_histogram(case):
    counts = torch.bincount(
        case.topk_ids.cpu().long().flatten(), minlength=case.num_experts
    )
    local = counts[
        case.local_expert_offset : case.local_expert_offset + case.local_num_experts
    ]
    return {
        "global": counts.tolist(),
        "local": local.tolist(),
        "local_assignments": int(local.sum()),
        "empty_local_experts": int((local == 0).sum()),
    }


def parallel_routing_histogram(topk_ids, num_experts, ep_size):
    """Global and per-rank assignment counts for uniform EP rank intervals.

    Under MoE tensor parallelism every rank sees the global histogram, so
    callers pass ``ep_size=1`` and the single per-rank entry equals it.
    """
    if num_experts % ep_size:
        raise ValueError("num_experts must be divisible by ep_size")
    counts = torch.bincount(topk_ids.cpu().long().flatten(), minlength=num_experts)
    local = num_experts // ep_size
    ranks = counts.reshape(ep_size, local)
    return {
        "global": counts.tolist(),
        "ep_size": ep_size,
        "per_rank": ranks.tolist(),
        "per_rank_assignments": ranks.sum(dim=1).tolist(),
        "per_rank_empty_experts": (ranks == 0).sum(dim=1).tolist(),
        "remote_fraction_per_rank": [
            float(1.0 - int(ranks[rank].sum()) / max(1, int(counts.sum())))
            for rank in range(ep_size)
        ],
    }


def situ_reference(up, gate, beta, linear_beta):
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return gate * up


@torch.no_grad()
def reference_moe(case: MXFP4SiTUCase, modes=("ideal_fp64", "mxfp8_fp32")):
    """Return FP64 rank-local output(s), dequantizing one expert at a time.

    ideal_fp64: both matmuls, SiTU and final weighted sum in FP64, no middle
    rounding. mxfp8_fp32: round the FP64 SiTU result to FP32 then MXFP8 before
    GEMM2. mxfp8_bf16 additionally rounds to BF16, matching the historical
    FlashInfer *test reference*, not asserting that fused kernels do so.
    FP64 is an oracle for arithmetic on the exact supplied quantized operands;
    it cannot reconstruct information lost before those operands were stored.
    """
    allowed = {"ideal_fp64", "mxfp8_fp32", "mxfp8_bf16"}
    if not set(modes) <= allowed:
        raise ValueError(f"reference modes must be in {allowed}")
    device = case.x.device
    x = decode_mxfp8(case.x, case.x_scale)
    outputs = {
        mode: torch.zeros(x.shape, dtype=torch.float64, device=device) for mode in modes
    }
    ids = case.topk_ids.cpu()
    weights = case.topk_weights.to(device=device, dtype=torch.float64)
    for local in range(case.local_num_experts):
        pairs = (ids == local + case.local_expert_offset).nonzero()
        if pairs.shape[0] == 0:
            continue
        token = pairs[:, 0].to(device)
        slot = pairs[:, 1].to(device)
        w1 = decode_mxfp4(case.w1[local], case.w1_scale[local], device=device)
        fc1 = x[token] @ w1.t()
        del w1
        up, gate = fc1.chunk(2, dim=-1)
        beta = case.beta[0 if case.beta.numel() == 1 else local].to(
            device=device, dtype=torch.float64
        )
        linear_beta = (
            None
            if case.linear_beta is None
            else case.linear_beta[0 if case.linear_beta.numel() == 1 else local].to(
                device=device, dtype=torch.float64
            )
        )
        activation = situ_reference(up, gate, beta, linear_beta)
        w2 = decode_mxfp4(case.w2[local], case.w2_scale[local], device=device)
        for mode in modes:
            middle = activation
            if mode != "ideal_fp64":
                source = (
                    activation.to(torch.bfloat16)
                    if mode == "mxfp8_bf16"
                    else activation.float()
                )
                q, sf = quantize_mxfp8_reference(source)
                middle = decode_mxfp8(q, sf)
            contribution = (middle @ w2.t()) * weights[token, slot, None]
            # Routing IDs are unique within a row: one expert updates any token
            # at most once, and experts are visited in a fixed order.
            outputs[mode][token] += contribution
        del w2, fc1, activation
    return outputs


def absolute_error_quantiles(absolute):
    """Exact linear p50/p95/p99, including arrays above torch.quantile's limit."""
    flat = absolute.flatten()
    probabilities = torch.tensor(
        [0.5, 0.95, 0.99], device=flat.device, dtype=torch.float64
    )
    if flat.numel() <= 2**24:
        return torch.quantile(flat, probabilities)
    # torch.quantile rejects >2**24 elements. Sorting itself has no such
    # restriction; compute its double-precision ranks and linear interpolation
    # explicitly. This runs only in diagnostics, never in the kernel path.
    ordered = flat.sort().values
    ranks = probabilities * (flat.numel() - 1)
    lower, upper = ranks.floor().long(), ranks.ceil().long()
    return torch.lerp(ordered[lower], ordered[upper], ranks - lower)


def error_metrics(actual, reference):
    """Diagnostics, with no customer-unapproved numerical pass threshold."""
    actual, reference = actual.double(), reference.double()
    finite = bool(torch.isfinite(actual).all() & torch.isfinite(reference).all())
    if not finite:
        return {"finite": False}
    error = actual - reference
    absolute = error.abs()
    ref_norm = torch.linalg.vector_norm(reference)
    actual_norm = torch.linalg.vector_norm(actual)
    error_norm = torch.linalg.vector_norm(error)
    relative_l2 = error_norm / ref_norm if ref_norm > 0 else error_norm
    if ref_norm > 0 and actual_norm > 0:
        cosine = (actual.flatten() @ reference.flatten()) / (ref_norm * actual_norm)
    else:
        cosine = (
            torch.ones_like(ref_norm) if error_norm == 0 else torch.zeros_like(ref_norm)
        )
    quantiles = absolute_error_quantiles(absolute)
    passing = absolute <= (0.1 + 0.15 * reference.abs())
    token_error = torch.linalg.vector_norm(error, dim=-1)
    token_norm = torch.linalg.vector_norm(reference, dim=-1)
    token_relative = torch.where(token_norm > 0, token_error / token_norm, token_error)
    return {
        "finite": True,
        "relative_l2": float(relative_l2),
        "cosine": float(cosine),
        "mean_abs": float(absolute.mean()),
        "p50_abs": float(quantiles[0]),
        "p95_abs": float(quantiles[1]),
        "p99_abs": float(quantiles[2]),
        "max_abs": float(absolute.max()),
        "worst_token_relative_l2": float(token_relative.max()),
        "fraction_atol_0.1_rtol_0.15": float(passing.double().mean()),
    }


def paired_accuracy(candidate, baseline, reference):
    candidate_metrics = error_metrics(candidate, reference)
    baseline_metrics = error_metrics(baseline, reference)
    result = {
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
        "candidate_vs_baseline": error_metrics(candidate, baseline),
    }
    if candidate_metrics["finite"] and baseline_metrics["finite"]:
        result["relative_l2_delta"] = (
            candidate_metrics["relative_l2"] - baseline_metrics["relative_l2"]
        )
        result["cosine_delta"] = (
            candidate_metrics["cosine"] - baseline_metrics["cosine"]
        )
    return result


@torch.no_grad()
def make_case(
    tokens=16,
    hidden=256,
    intermediate=128,
    num_experts=8,
    local_num_experts=4,
    local_expert_offset=2,
    top_k=2,
    distribution="balanced",
    seed=123,
    beta=4.0,
    linear_beta=25.0,
    input_distribution="normal",
    device="cuda",
):
    """Generate quantized fixtures with bounded GPU weight staging.

    Canonical weight banks live on CPU; one expert is generated/quantized at a
    time on the GPU. Prepared backends can discard their temporary canonical
    GPU bank once shuffled. No model download is involved.
    """
    from flashinfer.fp4_quantization import fp4_quantize

    generator = torch.Generator(device=device).manual_seed(seed)
    x_source = torch.randn((tokens, hidden), device=device, generator=generator)
    if input_distribution == "outliers":
        x_source[:, ::97] *= 16
    elif input_distribution == "small":
        x_source *= 0.01
    elif input_distribution == "saturated":
        x_source *= 16
    elif input_distribution == "zero":
        x_source.zero_()
    elif input_distribution != "normal":
        raise ValueError(f"unknown input distribution: {input_distribution}")
    x, x_sf = quantize_mxfp8_reference(x_source)
    banks = []
    for rows, columns in ((2 * intermediate, hidden), (hidden, intermediate)):
        packed = torch.empty((local_num_experts, rows, columns // 2), dtype=torch.uint8)
        scales = torch.empty(
            (local_num_experts, rows, columns // 32), dtype=torch.uint8
        )
        for expert in range(local_num_experts):
            dense = (
                torch.randn((rows, columns), device=device, generator=generator)
                / columns**0.5
            ).to(torch.bfloat16)
            q, sf = fp4_quantize(
                dense,
                global_scale=torch.ones(1, device=device),
                sf_vec_size=32,
                sf_use_ue8m0=True,
                is_sf_swizzled_layout=False,
            )
            packed[expert].copy_(q.cpu())
            scales[expert].copy_(
                sf.view(torch.uint8).reshape(rows, columns // 32).cpu()
            )
        banks.extend((packed, scales))
    ids, weights = make_routing(
        tokens,
        num_experts,
        top_k,
        local_num_experts,
        local_expert_offset,
        distribution,
        device=device,
        seed=seed + 1,
    )
    betas = torch.full((local_num_experts,), beta, dtype=torch.float32, device=device)
    linear_betas = None if linear_beta is None else torch.full_like(betas, linear_beta)
    return MXFP4SiTUCase(
        x,
        x_sf,
        *banks,
        ids,
        weights,
        betas,
        linear_betas,
        num_experts,
        local_expert_offset,
    )


@torch.no_grad()
def prepare_cute_weights(case):
    """Shuffle supplied native bytes; never dequantize or requantize weights."""
    from flashinfer.fused_moe.prepare import prepare_cute_dsl_mxfp4_weights

    return prepare_cute_dsl_mxfp4_weights(
        case.w1.to(case.x.device),
        case.w1_scale.to(case.x.device),
        case.w2.to(case.x.device),
        case.w2_scale.to(case.x.device),
    )


@torch.no_grad()
def prepare_trt_weights(case):
    """Build TRT physical layouts from the identical canonical packed bank."""
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )
    from flashinfer.quantization.fp4_quantization import block_scale_interleave

    device = case.x.device
    cache, result = {}, []
    for packed, scales, first in (
        (case.w1, case.w1_scale, True),
        (case.w2, case.w2_scale, False),
    ):
        q = torch.empty(packed.shape, dtype=torch.uint8, device=device)
        sf = torch.empty(scales.shape, dtype=torch.uint8, device=device)
        permute = (
            _maybe_get_cached_w3_w1_permute_indices
            if first
            else get_w2_permute_indices_with_cache
        )
        for expert in range(packed.shape[0]):
            expert_q, expert_sf = packed[expert].to(device), scales[expert].to(device)
            p = permute(cache, expert_q, 128, is_gated_act_gemm=True).to(device)
            p_sf = permute(
                cache, expert_sf, 128, num_elts_per_sf=16, is_gated_act_gemm=True
            ).to(device)
            q[expert].copy_(expert_q[p])
            sf[expert].copy_(
                block_scale_interleave(expert_sf[p_sf].contiguous()).reshape_as(
                    expert_sf
                )
            )
        result.extend((q, sf.view(torch.float8_e4m3fn)))
    return tuple(result)


def make_trt_baseline(case, prepared_weights=None, *, packed=True):
    """Prepare the explicit TRT-LLM Gen baseline outside timed execution."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.tllm_enums import ActivationType, SfLayout

    if case.linear_beta is None:
        raise ValueError("TRT's absent gemm1_beta means 1, not disabled up-branch tanh")
    w1, s1, w2, s2 = (
        prepare_trt_weights(case) if prepared_weights is None else prepared_weights
    )
    ones = torch.ones(case.local_num_experts, device=case.x.device, dtype=torch.float32)
    routes = (
        pack_topk(case.topk_ids, case.topk_weights)
        if packed
        else (case.topk_ids, case.topk_weights)
    )
    output = torch.empty_like(case.x, dtype=torch.bfloat16)
    kwargs = dict(
        topk_ids=routes,
        routing_bias=None,
        hidden_states=case.x,
        hidden_states_scale=case.x_scale.view(torch.float8_e4m3fn),
        hidden_states_scale_layout=SfLayout.layout_linear,
        gemm1_weights=w1,
        gemm1_weights_scale=s1,
        gemm1_bias=None,
        gemm1_alpha=case.beta,
        gemm1_beta=case.linear_beta,
        gemm1_clamp_limit=None,
        gemm2_weights=w2,
        gemm2_weights_scale=s2,
        gemm2_bias=None,
        output1_scale_scalar=ones,
        output1_scale_gate_scalar=ones,
        output2_scale_scalar=ones,
        per_token_scale=None,
        num_experts=case.num_experts,
        top_k=case.topk_ids.shape[1],
        n_group=None,
        topk_group=None,
        intermediate_size=case.intermediate_size,
        local_expert_offset=case.local_expert_offset,
        local_num_experts=case.local_num_experts,
        routed_scaling_factor=1.0,
        routing_method_type=0,
        do_finalize=True,
        enable_pdl=False,
        activation_type=ActivationType.Situ,
        output=output,
        tune_max_num_tokens=max(8192, case.x.shape[0]),
    )

    def run():
        trtllm_fp4_block_scale_routed_moe(**kwargs)
        return output

    return run, output
