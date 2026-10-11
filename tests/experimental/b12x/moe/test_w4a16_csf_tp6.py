"""W4A16 scale expansion preserves native arithmetic for partial TP tiles."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.moe._shared.kernels.w4a16.kernel import _select_tile_config
from b12x.preparation import PreparationSession, PreparedCall
from ..conftest import require_b12x

_E2M1 = (
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
)


@pytest.mark.parametrize("block", [8, 32, 64])
def test_k32_down_tiles_pair_with_n64_gate_up_tiles(block):
    """GLM-5.3 TP6: FC1 N = 704 and FC2 K = 352 share a 128-thread geometry."""
    common = dict(
        moe_block_size=block,
        sms=188,
        max_shared_mem=101_376,
        scale_format="e4m3_k16",
        weight_layout="packed",
    )
    fc1 = _select_tile_config(
        problem_m=4, problem_n=704, problem_k=6144, top_k=8, **common
    )
    fc2 = _select_tile_config(
        problem_m=32, problem_n=6144, problem_k=352, top_k=1, **common
    )
    assert fc1[:3] == (128, 64, 128)
    assert fc2[:3] == (32, 256, 128)
    # K32 stays a fallback: a K that K64 tiles divide never selects it.
    assert (
        _select_tile_config(
            problem_m=32, problem_n=6144, problem_k=256, top_k=1, **common
        )[0]
        >= 64
    )


def _logical_scales(rng, experts, rows, columns, outliers):
    base = rng.integers(0x28, 0x48, size=(experts, rows, 1))
    scales = base + rng.integers(0, 14, size=(experts, rows, columns))
    hot = rng.random((experts, rows, columns)) < outliers
    scales[hot] = rng.integers(0x40, 0x7E, size=int(hot.sum()))
    return scales.astype(np.uint8)


def _swizzled(logical, device):
    """F8_128x4 storage of logical ``[E, rows, columns]`` scales, padded."""
    e, rows, columns = logical.shape
    pr, pc = -(-rows // 128) * 128, -(-columns // 4) * 4
    padded = np.zeros((e, pr, pc), dtype=np.uint8)
    padded[:, :rows, :columns] = logical
    order = padded.reshape(e, pr // 128, 4, 32, pc // 4, 4).transpose(0, 1, 4, 3, 2, 5)
    return (
        torch.from_numpy(np.ascontiguousarray(order).reshape(e, pr, pc))
        .to(device)
        .view(torch.float8_e4m3fn)
    )


def _planes(logical):
    """Canonical byte-window CSF planes: 16-row slabs of bases and 4-bit codes."""
    fixed, exceptions = [], []
    for plane in logical:
        rows = plane.shape[0]
        base = np.minimum(plane.min(axis=1), 240).astype(np.uint8)
        offsets = plane.astype(np.int16) - base[:, None]
        outside = offsets > 15
        offsets[outside] = 0
        offsets = offsets.astype(np.uint8)
        packed = offsets[:, ::2] | (offsets[:, 1::2] << 4)
        fixed.append(
            np.concatenate((base.reshape(-1, 16), packed.reshape(rows // 16, -1)), 1)
        )
        position = np.flatnonzero(outside).astype(np.uint32)
        exceptions.append(position | (plane.ravel()[position].astype(np.uint32) << 24))
    return fixed, exceptions


def _dequant(weights, logical):
    lut = torch.tensor(_E2M1, device=weights.device)
    values = torch.stack((lut[(weights & 15).long()], lut[(weights >> 4).long()]), -1)
    values = values.reshape(weights.shape[0], weights.shape[1], -1)
    scales = (
        torch.from_numpy(logical).to(weights.device).view(torch.float8_e4m3fn).float()
    )
    return values * scales.repeat_interleave(16, dim=2)


@pytest.mark.parametrize(
    "n,raw_planes",
    [
        (96, True),
        (352, True),
        (128, False),
    ],
)
def test_tp6_shard_native_and_csf_match_the_oracle(n, raw_planes, monkeypatch):
    device = require_b12x()
    from b12x.moe.fused_moe import _impl

    e, h, topk = 8, 256, 2
    rng = np.random.default_rng(n)
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

    def packed(s13, s2):
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

    def compressed(logical):
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

    native = moe.prepare_weights(
        plan=plan, weights=packed(_swizzled(l13, device), _swizzled(l2, device))
    )
    buffers = (_swizzled(l13, device), _swizzled(l2, device))
    csf = moe.prepare_weights(
        plan=plan,
        weights=moe.Nvfp4CsfWeights(
            packed=packed(*buffers),
            w13_scales=compressed(l13),
            w2_scales=compressed(l2),
        ),
    )
    assert csf.plan._impl.w4a16_scale_format == ("e4m3_k16")
    stored = csf._impl.w1_blockscale
    assert stored.numel() > 0
    r13, r2 = _dequant(w13, l13), _dequant(w2, l2)
    for tokens in (1, 4, 33, 300):
        x = (torch.randn(tokens, h, device=device) * 0.5).to(torch.bfloat16)
        ids = torch.stack(
            [torch.randperm(e, device=device)[:topk] for _ in range(tokens)]
        ).to(torch.int32)
        weights = torch.softmax(
            torch.randn(tokens, topk, device=device), dim=-1
        ).float()
        plans = [
            moe.plan_execution(
                experts=owner,
                capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=topk),
                invocation={"fast_math": True},
            )
            for owner in (native, csf)
        ]

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

        outputs = []
        with PreparationSession(
            device=device, autotune=False, compile_workers=0
        ) as session:
            session.prepare(
                tuple(
                    p.request(name=f"tp6-{n}-{tokens}-{i}", prepare_call=prepare)
                    for i, p in enumerate(plans)
                )
            )
            for p in plans:
                scratch = tuple(
                    torch.empty(s.shape, dtype=s.dtype, device=device)
                    for s in p.scratch_specs()
                )
                output = torch.full_like(x, float("nan"))
                moe.run(
                    binding=moe.bind(
                        p,
                        a=x,
                        topk_ids=ids,
                        topk_weights=weights,
                        output=output,
                        scratch=scratch,
                        input_scales_static=True,
                    )
                )
                outputs.append(output)
            torch.cuda.synchronize()
        reference = torch.zeros(tokens, h, device=device)
        hidden = torch.einsum("rk,tk->tr", r13.reshape(-1, h), x.float()).reshape(
            tokens, e, 2 * n
        )
        for slot in range(topk):
            rows = hidden[torch.arange(tokens), ids[:, slot].long()]
            # W13 rows arrive up then gate.
            act = (
                (torch.nn.functional.silu(rows[:, n:]) * rows[:, :n])
                .to(torch.bfloat16)
                .float()
            )
            down = torch.einsum("thk,tk->th", r2[ids[:, slot].long()], act)
            reference += weights[:, slot : slot + 1] * down
        assert torch.isfinite(outputs[0]).all() and torch.count_nonzero(outputs[0])
        cosine = torch.nn.functional.cosine_similarity(
            outputs[0].float().flatten(), reference.flatten(), dim=0
        )
        assert cosine > 0.9999, (tokens, float(cosine))
        torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
