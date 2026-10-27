"""Compressed and native W4A16 weights use the same cooperative compute grid."""

import numpy as np
import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall
from .test_w4a16_csf_tp6 import _logical_scales, _planes, _swizzled
from ..conftest import require_b12x

_CAPACITIES = tuple(range(1, 257)) + (8192,)


@pytest.mark.parametrize("intermediate", [256, 352])
def test_csf_plans_fit_their_cooperative_grid_at_every_capacity(
    intermediate, monkeypatch
):
    device = require_b12x()
    monkeypatch.setenv("B12X_W4A16_SMALL_M_OCCUPANCY", "2")
    e, h, n, topk = 256, 6144, intermediate, 8
    rng = np.random.default_rng(intermediate)
    w13 = torch.randint(0, 256, (e, 2 * n, h // 2), dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (e, h, n // 2), dtype=torch.uint8, device=device)
    l13 = _logical_scales(rng, e, 2 * n, h // 16, 0.03)
    l2 = _logical_scales(rng, e, h, n // 16, 0.03)
    plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
    )
    one = torch.ones(e, device=device)

    def packed(s13, s2, w13, w2):
        return moe.PackedWeights(
            w13=w13.clone(),
            w2=w2.clone(),
            w13_block_scales=s13,
            w2_block_scales=s2,
            w13_global_scales=one,
            w2_global_scales=one,
            input_scale=one,
            intermediate_scale=one,
            immutable_input_scales=True,
        )

    def planes(logical):
        from b12x.loader._csf_blocks import export_csf_blocks

        fixed, exceptions = _planes(logical)
        rows, columns = logical.shape[1:]
        return moe.CsfBlockPlanes(
            tuple(
                export_csf_blocks(
                    f, x, rows=rows, columns=columns, quantization="nvfp4"
                )
                for f, x in zip(fixed, exceptions, strict=True)
            )
        )

    owners = (
        moe.prepare_weights(
            plan=plan,
            weights=packed(_swizzled(l13, device), _swizzled(l2, device), w13, w2),
        ),
        moe.prepare_weights(
            plan=plan,
            weights=moe.Nvfp4CsfWeights(
                packed=packed(_swizzled(l13, device), _swizzled(l2, device), w13, w2),
                w13_scales=planes(l13),
                w2_scales=planes(l2),
            ),
        ),
    )
    del w13, w2
    assert owners[1].plan._impl.w4a16_scale_format == ("e4m3_k16")
    torch.manual_seed(intermediate)
    x_all = (torch.randn(max(_CAPACITIES), h, device=device) * 0.5).to(torch.bfloat16)
    ids_all = torch.stack(
        [torch.randperm(e, device=device)[:topk] for _ in range(max(_CAPACITIES))]
    ).to(torch.int32)
    weights_all = torch.softmax(
        torch.randn(max(_CAPACITIES), topk, device=device), dim=-1
    ).float()

    def inputs(tokens):
        return x_all[:tokens], ids_all[:tokens], weights_all[:tokens]

    def prepare_for(tokens):
        x, ids, weights = inputs(tokens)

        def prepare(state):
            scratch = tuple(
                torch.empty(s.shape, dtype=s.dtype, device=device)
                for s in state.scratch.scratch_specs()
            )
            output = torch.empty_like(x)
            binding = state.bind(
                a=x,
                topk_ids=ids,
                topk_weights=weights,
                output=output,
                scratch=scratch,
                input_scales_static=True,
            )
            return PreparedCall(
                run=lambda: state.run(binding), output=output, owners=(scratch, binding)
            )

        return prepare

    plans = {
        tokens: [
            moe.plan_execution(
                experts=owner,
                capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=topk),
                invocation={"fast_math": True},
            )
            for owner in owners
        ]
        for tokens in _CAPACITIES
    }
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        # Preparation primes every plan with a launch: an oversized cooperative
        # grid fails here.
        session.prepare(
            tuple(
                p.request(
                    name=f"csf-residency-{n}-{tokens}-{i}",
                    prepare_call=prepare_for(tokens),
                )
                for tokens, pair in plans.items()
                for i, p in enumerate(pair)
            )
        )
        for tokens, pair in plans.items():
            x, ids, weights = inputs(tokens)
            outputs, residency = [], []
            for p in pair:
                scratch = tuple(
                    torch.empty(s.shape, dtype=s.dtype, device=device)
                    for s in p.scratch_specs()
                )
                output = torch.full_like(x, float("nan"))
                binding = moe.bind(
                    p,
                    a=x,
                    topk_ids=ids,
                    topk_weights=weights,
                    output=output,
                    scratch=scratch,
                    input_scales_static=True,
                )
                launch = getattr(binding, "fused_launch", None)
                residency.append(None if launch is None else launch.blocks_per_sm)
                moe.run(binding=binding)
                outputs.append(output)
            torch.cuda.synchronize()
            assert torch.isfinite(outputs[0]).all() and torch.count_nonzero(
                outputs[0]
            ), tokens
            assert torch.isfinite(outputs[1]).all(), tokens
            scale = float(outputs[0].float().abs().max())
            assert residency[0] == residency[1], (tokens, residency)
            # Both scale formats run the same schedule. Its cross-CTA
            # split-K reductions (and the prefill route sum) are not
            # ordered: native differs from itself by a BF16 rounding in
            # rare elements, so allow a few such elements.
            differ = outputs[1] != outputs[0]
            rare = outputs[0].numel() // (10_000 if tokens <= 256 else 1_000)
            assert int(differ.sum()) <= max(16, rare), tokens
            if bool(differ.any()):
                delta = (outputs[1].float() - outputs[0].float()).abs()[differ]
                assert float(delta.max()) <= 2.0**-6 * scale, tokens
