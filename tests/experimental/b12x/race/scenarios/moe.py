"""Fused MoE race scenarios (b12x W4A16, dynamic and micro kernels).

Every family builds seeded synthetic experts, prepares one exact-M execution
of ``fused_moe`` (the serving entry point) in a frozen ``PreparationSession``,
and replays ``fused_moe.run`` into a NaN-poisoned output. The kernels under
test synchronize CTAs through software grid barriers, so each family checks
that the prepared binding really selected the implementation it targets and
that an eager run writes finite, nonzero output (a scenario whose output were
all NaN would replay "bit-exactly" and prove nothing).

- ``moe_w4a16``: the W4A16 fused kernel. M >= 9 (and smaller M with enough
  route reuse) runs the route-packed cooperative body (FC1, activation and FC2
  phases separated by ``_grid_barrier`` over ``sms * blocks_per_sm`` resident
  CTAs, ordered split-K turns) followed by a fixed-order top-k sum; other
  M <= 8 runs the same fused kernel on direct (unpacked) top-k routes. Cases
  keep at most ~32 routes per expert, since larger packs are not reproducible
  even on a quiet GPU (see ``_fuzz_moe_w4a16``).
- ``moe_dynamic``: the persistent cooperative dynamic kernel (NVFP4, W4A8 or
  W8A8 weights) with its atomic task queue and ``_resident_grid_barrier``,
  always with ``deterministic_output=True`` (one row per routed pair, then a
  fixed-order top-k sum). The default reduction, a BF16 atomic scatter of one
  FC2 partial per intermediate slice, differs between quiet replays even at
  top-k 1, so it cannot be checked bit-exactly.
- ``moe_micro``: the cooperative decode micro kernel (M <= 8 tokens, fewer
  than 64 routed rows). M = 1 hands FC1 to FC2 through the per-token
  ``_token_publish_fc1_ready``/``_token_wait_fc1_ready`` pair, whose arrival
  atomic is relaxed with no fence after the CTA sync; M >= 2 uses the fenced
  ``_resident_grid_barrier``.
"""

from __future__ import annotations

import random

import torch

from ._base import Scenario, _generator


def _routes(gen, m, num_experts, top_k, device):
    """Distinct experts per token and softmax weights, drawn from ``gen``."""
    ids = torch.rand((m, num_experts), generator=gen).argsort(dim=1)[:, :top_k]
    weights = torch.softmax(torch.randn((m, top_k), generator=gen), dim=-1)
    return (
        ids.to(torch.int32).contiguous().to(device),
        weights.to(torch.float32).contiguous().to(device),
    )


def _nvfp4_experts(gen, case, a, mode):
    """ModelOpt NVFP4 experts with signed random codes and per-block scales,
    prepared for ``mode`` (``w4a16``, ``nvfp4`` or ``w4a8_nvfp4``)."""
    from b12x._lib.intrinsics import swizzle_block_scale
    from b12x.testing.reference.helpers import prepare_tp_moe_fp4_experts

    device = a.device
    e, k, n = case["experts"], case["hidden"], case["intermediate"]

    def codes(*shape):
        return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8).to(device)

    def scales(*shape):
        values = torch.rand(shape, generator=gen) * 0.25 + 0.03125
        return swizzle_block_scale(values.to(device).to(torch.float8_e4m3fn))

    def alphas():
        return (torch.rand((e,), generator=gen) * 0.1 + 0.05).to(device)

    w13, w13_scale, w13_alpha = (
        codes(e, 2 * n, k // 2),
        scales(e, 2 * n, k // 16),
        alphas(),
    )
    w2, w2_scale, w2_alpha = codes(e, k, n // 2), scales(e, k, n // 16), alphas()
    unit = torch.ones(e, dtype=torch.float32, device=device)
    return prepare_tp_moe_fp4_experts(
        a=a,
        a1_gscale=unit,
        w1_fp4=w13,
        w1_blockscale=w13_scale,
        w1_alphas=w13_alpha,
        a2_gscale=unit,
        w2_fp4=w2,
        w2_blockscale=w2_scale,
        w2_alphas=w2_alpha,
        activation="silu",
        quant_mode=mode,
    )


def _mxfp8_experts(gen, case, device):
    """MXFP8 (E4M3 codes, UE8M0 K/32 scales) experts quantized from BF16 rows,
    as in test_mxfp8_w8a8.py, with distinct per-expert global scales."""
    from b12x.gemm._shared.wo_mxfp8 import quantize_mxfp8_rows_torch
    from b12x.moe import fused_moe

    e, k, n = case["experts"], case["hidden"], case["intermediate"]

    def quantized(rows, cols):
        padded = -(-cols // 128) * 128
        source = (
            (torch.randn((rows, cols), generator=gen) * 0.2)
            .to(torch.bfloat16)
            .to(device)
        )
        q = quantize_mxfp8_rows_torch(
            torch.nn.functional.pad(source, (0, padded - cols))
        )
        return (
            q.values.view(torch.uint8)[:, :cols].contiguous(),
            q.scale_rows.reshape(rows, padded // 32)
            .view(torch.uint8)[:, : cols // 32]
            .contiguous(),
        )

    w13 = [quantized(2 * n, k) for _ in range(e)]
    w2 = [quantized(k, n) for _ in range(e)]
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat.MXFP8_E8M0_K32,
            w13_layout=fused_moe.W13Layout.W13,
        ),
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A8,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=e, hidden_size=k, intermediate_size=n
        ),
    )

    def ramp(lo, hi):
        return torch.linspace(lo, hi, e, dtype=torch.float32, device=device)

    return fused_moe.prepare_weights(
        plan=plan,
        weights=fused_moe.PackedWeights(
            w13=torch.stack([v for v, _ in w13]),
            w2=torch.stack([v for v, _ in w2]),
            w13_block_scales=torch.stack([s for _, s in w13]),
            w2_block_scales=torch.stack([s for _, s in w2]),
            w13_global_scales=ramp(0.25, 0.75),
            w2_global_scales=ramp(1.5, 3.0),
            input_scale=ramp(0.5, 2.0),
            intermediate_scale=ramp(4.0, 6.0),
            immutable_input_scales=True,
        ),
    )


def _fused_moe(device, case, *, mode, implementation, deterministic) -> Scenario:
    """Prepare and freeze one exact-M ``fused_moe`` execution (the flow of
    ``b12x.testing.reference.helpers.make_tp_moe_fp4_binding``, kept open)."""
    from b12x.moe import fused_moe
    from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall
    from b12x.preparation.types import require_prepared

    m, top_k = case["m"], case["top_k"]
    gen = _generator(case)
    a = (
        (torch.randn((m, case["hidden"]), generator=gen) * 0.25)
        .to(torch.bfloat16)
        .to(device)
    )
    if mode == "w8a8_mx":
        experts = _mxfp8_experts(gen, case, device)
    else:
        experts = _nvfp4_experts(gen, case, a, mode)
    topk_ids, topk_weights = _routes(gen, m, case["experts"], top_k, device)
    output = torch.empty_like(a)

    declaration = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=m, top_k=top_k),
        routing=fused_moe.RoutingSpec(deterministic_output=deterministic),
        invocation=FrozenMapping({"fast_math": True}),
    )
    tensors = dict(
        a=a,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        input_scales_static=True,
    )

    def scratch_for(state):
        return tuple(
            torch.zeros(spec.shape, dtype=spec.dtype, device=spec.device)
            for spec in state.scratch.scratch_specs()
        )

    def prepare_call(state):
        scratch = scratch_for(state)
        binding = state.bind(scratch=scratch, **tensors)
        return PreparedCall(run=lambda: state.run(binding), owners=scratch)

    request = declaration.request(
        name=f"race.moe_{implementation}", prepare_call=prepare_call
    )
    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    session.__enter__()
    try:
        session.prepare((request,))
        session.freeze()
        scratch = scratch_for(require_prepared(request.plan, "moe.decode"))
        binding = fused_moe.bind(request.plan, scratch=scratch, **tensors)
        selected = getattr(binding, "implementation", None)
        if selected != implementation:
            raise ValueError(
                f"case {case} selected the {selected!r} MoE kernel, not {implementation!r}"
            )

        def forward():
            fused_moe.run(binding=binding)

        output.fill_(float("nan"))
        forward()
        torch.cuda.synchronize(device)
        if not bool(torch.isfinite(output).all()) or not bool(output.ne(0).any()):
            raise ValueError(f"case {case}: eager output is not finite and nonzero")
    except BaseException:
        session.__exit__(None, None, None)
        raise

    return Scenario(
        forward=forward,
        outputs={"out": output},
        reset=lambda: output.fill_(float("nan")),
        close=lambda: session.__exit__(None, None, None),
        keepalive=[
            session,
            request,
            binding,
            scratch,
            experts,
            a,
            topk_ids,
            topk_weights,
        ],
        description=(
            f"fused MoE {implementation} {mode} M={m} E={case['experts']} top_k={top_k} "
            f"{case['hidden']}->{case['intermediate']}"
            + (", deterministic output" if deterministic else "")
        ),
    )


def moe_w4a16(device, case) -> Scenario:
    """W4A16 fused MoE: the route-packed grid-barrier body, or small-M direct."""
    return _fused_moe(
        device, case, mode="w4a16", implementation="w4a16", deterministic=False
    )


def moe_dynamic(device, case) -> Scenario:
    """Persistent dynamic kernel (``mode`` nvfp4, w4a8_nvfp4 or w8a8_mx) with
    the fixed-order deterministic output reduction."""
    return _fused_moe(
        device,
        case,
        mode=case.get("mode", "nvfp4"),
        implementation="dynamic",
        deterministic=True,
    )


def moe_micro(device, case) -> Scenario:
    """Cooperative decode micro kernel (M <= 8, fewer than 64 routed rows)."""
    return _fused_moe(
        device,
        case,
        mode=case.get("mode", "nvfp4"),
        implementation="micro",
        deterministic=False,
    )


FAMILIES = {
    "moe_w4a16": moe_w4a16,
    "moe_dynamic": moe_dynamic,
    "moe_micro": moe_micro,
}

# GLM-style 16-expert top-8 shard (test_fused_moe's M=9 grid-barrier shape),
# a 64-expert top-8 layer, a Qwen-style top-10 MTP expert shape, and a
# small-intermediate top-4 layer for many short tiles.
_GLM16 = {"experts": 16, "hidden": 6144, "intermediate": 512, "top_k": 8}
_E64 = {"experts": 64, "hidden": 2048, "intermediate": 768, "top_k": 8}
_QWEN = {"experts": 16, "hidden": 2560, "intermediate": 640, "top_k": 10}
_E32 = {"experts": 32, "hidden": 2048, "intermediate": 512, "top_k": 4}

CASES = {
    "moe_w4a16": [
        {**_GLM16, "m": 9},
        {**_GLM16, "m": 64},
        {**_E64, "m": 33},
        {**_E32, "m": 256},
        {**_GLM16, "m": 8},  # packed: 64 routes over 16 experts reuse rows
        {**_E64, "m": 1},  # direct top-k routes
        {**_E64, "m": 4},  # direct top-k routes
    ],
    "moe_dynamic": [
        {**_E64, "m": 64, "mode": "nvfp4"},
        {**_E64, "m": 256, "mode": "nvfp4"},
        {**_E32, "m": 512, "mode": "nvfp4"},
        {**_E64, "m": 64, "mode": "w4a8_nvfp4"},
        {**_QWEN, "m": 64, "mode": "w8a8_mx"},
    ],
    "moe_micro": [
        {**_E64, "m": 1, "mode": "nvfp4"},
        {**_GLM16, "m": 1, "mode": "nvfp4"},
        {**_E64, "m": 4, "mode": "nvfp4"},
        {**_E64, "m": 7, "mode": "nvfp4"},
        {**_E64, "m": 1, "mode": "w4a8_nvfp4"},
        {**_E64, "m": 4, "mode": "w4a8_nvfp4"},
    ],
}


def _fuzz_geometry(rng: random.Random) -> dict:
    experts = rng.choice((8, 16, 32, 64, 128))
    return {
        "experts": experts,
        "hidden": 256 * rng.randint(4, 24),
        "intermediate": 128 * rng.randint(2, 12),
        "top_k": rng.choice([t for t in (1, 2, 4, 6, 8, 10) if t <= experts]),
        "seed": rng.getrandbits(31),
    }


def _fuzz_moe_w4a16(rng: random.Random) -> dict:
    case = _fuzz_geometry(rng)
    # At about 64 or more routes per expert (an expert spanning several 64-row
    # route blocks) quiet replays already differ in a few bytes: the multi-CTA
    # relaxed-atomic route sort packs routes in a different order each replay
    # and the result depends on that order. Stay at or below 32 per expert.
    limit = 32 * case["experts"] // case["top_k"]
    case["m"] = rng.choice(
        [m for m in (9, 12, 17, 32, 48, 96, 160, 256) if m <= limit] or [9]
    )
    return case


def _fuzz_moe_dynamic(rng: random.Random) -> dict:
    case = _fuzz_geometry(rng)
    case["mode"] = rng.choice(("nvfp4", "nvfp4", "w4a8_nvfp4", "w8a8_mx"))
    if case["mode"] == "w8a8_mx":
        # Keep the per-expert BF16 quantization pass and expert memory small.
        case["experts"] = min(case["experts"], 32)
        case["top_k"] = min(case["top_k"], case["experts"])
    # Micro owns at most 8 tokens, so M >= 9 is always dynamic.
    case["m"] = rng.choice((9, 16, 33, 64, 128, 256, 512))
    return case


def _fuzz_moe_micro(rng: random.Random) -> dict:
    case = _fuzz_geometry(rng)  # K <= 6144 keeps micro's direct K segments <= 12
    case["mode"] = rng.choice(("nvfp4", "nvfp4", "w4a8_nvfp4"))
    # Micro owns M <= 8 with fewer than 64 routed rows (W4A8: at most 32).
    rows = 32 if case["mode"] == "w4a8_nvfp4" else 63
    case["top_k"] = min(case["top_k"], rows)
    case["m"] = rng.randint(1, max(1, min(8, rows // case["top_k"])))
    return case


FUZZ = {
    "moe_w4a16": _fuzz_moe_w4a16,
    "moe_dynamic": _fuzz_moe_dynamic,
    "moe_micro": _fuzz_moe_micro,
}


# local-inference-lab/flashinfer#18: the W4A16 grid barrier's relaxed arrival
# atomic let other CTAs pass before this CTA's FC1/activation stores were visible.
_W4A16_BARRIER_FENCE = (
    "            # The arrival atomic is relaxed: release this CTA's global writes, which\n"
    "            # sync_threads made visible to thread 0, before other CTAs can pass.\n"
    "            threadfence()\n"
)
_W4A16_MUTATION = {
    "module": "b12x.moe._shared.kernels.w4a16.kernel",
    "replace": [(_W4A16_BARRIER_FENCE, "")],
    "family": "moe_w4a16",
    "cases": [
        {**_GLM16, "m": 9},
        {**_GLM16, "m": 64},
        {**_E32, "m": 256},
        {**_E64, "m": 4},
    ],
}

MUTATIONS = {
    "w4a16_grid_barrier_fence": {**_W4A16_MUTATION, "delay": False},
    "w4a16_grid_barrier_fence_delay": {**_W4A16_MUTATION, "delay": True},
}
