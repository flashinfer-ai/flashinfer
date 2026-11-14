"""Check staged Kimi activation bytes against an independent FP64 oracle."""

import pytest
import torch

from flashinfer import moe_ep
from flashinfer.moe_ep.kernel_src.sm100 import _kimi_k3_megamoe as provider

pytestmark = [pytest.mark.arch_blackwell, pytest.mark.gpu_exclusive]

HIDDEN = 3584
FP4 = torch.tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6), dtype=torch.float64)
FP8 = torch.tensor(
    [
        (code & 7) * 2.0**-9
        if code < 8
        else (1 + (code & 7) / 8) * 2.0 ** ((code >> 3) - 7)
        for code in range(127)
    ],
    dtype=torch.float64,
)


def _encode(values, grid, sign_bit):
    """Choose the nearest finite code; an exact tie selects the even code."""
    distance = (values.abs().clamp(max=grid[-1])[..., None] - grid).abs()
    code = distance.argmin(-1)
    upper = (code + 1).clamp(max=grid.numel() - 1)
    tied = distance.gather(-1, code[..., None]) == distance.gather(-1, upper[..., None])
    code += (tied.squeeze(-1) & (upper > code) & (code % 2 == 1)).long()
    return code.byte() | (torch.signbit(values).byte() * sign_bit)


def _input(kind):
    block = 16 if kind == "nvfp4" else 32
    tiny = 2.0**-133  # Smallest nonzero BF16 value.
    cases = [
        [0.0, -0.0] * (block // 2),
        [tiny, -tiny, 2.0**-126, -(2.0**-126)] * (block // 4),
    ]
    if kind == "nvfp4":
        # A maximum of six fixes scale=norm, so payload ties stay exact.
        midpoints = ((FP4[:-1] + FP4[1:]) / 2).tolist()
        cases.append([6.0, -6.0, *midpoints, *[-v for v in midpoints]])
        # Scale ties: 1/1024 rounds to zero; 3/1024 rounds to 1/256.
        for maximum in (6.0 / 1024, 18.0 / 1024):
            cases.append([maximum, -maximum] * 8)
        cases.append([6, -6, 0, -0.0, 0.125, -0.125, 0.5, -0.5] * 2)
    else:
        # Cover FP8 subnormal and normal midpoint ties at unit block scale.
        midpoints = ((FP8[:-1] + FP8[1:]) / 2).tolist()
        for start in range(0, len(midpoints), 15):
            values = midpoints[start : start + 15]
            values += [1.0] * (15 - len(values))
            cases.append([448, -448, *values, *[-v for v in values]])
        cases.append([1, -2, 3.5, -5, 0, -0.0, 17, -29] * 4)
    pattern = torch.tensor(cases, dtype=torch.float64).flatten()
    values = pattern.repeat((2 * HIDDEN + pattern.numel() - 1) // pattern.numel())
    return values[: 2 * HIDDEN].reshape(2, HIDDEN).bfloat16()


def _reference(x, kind, norm):
    block = 16 if kind == "nvfp4" else 32
    values = x.double().reshape(2, -1, block)
    maximum = values.abs().amax(-1)
    if kind == "nvfp4":
        scales = _encode(maximum / 6 * norm, FP8, 128)
        inverse = torch.where(maximum == 0, 0.0, norm / FP8[scales.long()])
        codes = _encode(values * inverse[..., None], FP4, 8).flatten(1)
        payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
    else:
        exponent = torch.ceil(torch.log2(maximum / 448)).clamp(-127, 127)
        scales = (exponent + 127).byte()
        # The upstream zero-scale encoding uses a unit reciprocal.
        inverse = torch.where(scales == 0, 1.0, 2.0 ** (-exponent))
        inverse = torch.where(maximum == 0, 0.0, inverse)
        payload = _encode(values * inverse[..., None], FP8, 128).flatten(1)
    return payload, scales


@pytest.mark.parametrize(
    ("kind", "norm"), [("nvfp4", 1.0), ("nvfp4", 2.0), ("mxfp4_mxfp8", 1.0)]
)
def test_kimi_staged_quantizer_bytes(kind, norm):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    config = moe_ep.Sm100_KimiK3_MegaMoeConfig(
        quant_kind=kind,
        situ_beta=4.0,
        situ_linear_beta=25.0,
        input_norm_const=norm,
    )
    workspace = provider.create_workspace(
        config,
        moe_ep.FleetParams(
            num_experts=896, max_tokens_per_rank=2, token_hidden_size=HIDDEN
        ),
        rank=0,
        world_size=1,
        process_group=None,
    )
    try:
        x = _input(kind)
        tensors = moe_ep.MoEEpTensors(
            x.cuda(),
            torch.arange(16, dtype=torch.int32, device="cuda").repeat(2, 1),
            torch.ones(2, 16, dtype=torch.float32, device="cuda"),
        )
        workspace.stage_inputs(tensors, quantize_input=True)
        payload, scales = _reference(x, kind, norm)
        torch.testing.assert_close(workspace.activation.cpu(), payload, rtol=0, atol=0)
        torch.testing.assert_close(
            workspace.activation_sf.cpu(), scales, rtol=0, atol=0
        )
    finally:
        workspace.destroy()
