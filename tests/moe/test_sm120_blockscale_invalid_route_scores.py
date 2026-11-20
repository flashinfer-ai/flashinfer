# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from flashinfer.fused_moe.specialized import fp8_blockscale_moe_sm120 as moe


@pytest.fixture(scope="module")
def weights():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor, props.multi_processor_count) != (12, 0, 188):
        pytest.skip("requires validated SM120 device profile")
    w13 = torch.empty((256, 1024, 2048), dtype=torch.float8_e4m3fn, device=device)
    w2 = torch.empty((256, 2048, 512), dtype=w13.dtype, device=device)
    for w in (w13, w2):
        w.view(torch.uint8).fill_(0x38)  # e4m3fn encoding of +1
    s13 = torch.ones((256, 8, 16), device=device, dtype=torch.float32)
    s2 = torch.ones((256, 16, 4), device=device, dtype=torch.float32)
    s2[255].fill_(2.0)
    return w13, w2, [s13, s2]


@pytest.mark.parametrize("tokens", [1, 4, 16, 64])
@pytest.mark.parametrize("all_invalid", [False, True])
def test_omit_invalid_route_scores(weights, tokens, all_invalid, monkeypatch):
    monkeypatch.delenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", raising=False)
    w13, w2, scales = weights
    device = w13.device
    x = torch.zeros((tokens, 2048), device=device, dtype=torch.bfloat16)
    x[:, 0] = 1
    ids = [-1, 0, 256, 255, -2, 1, 257, 0]
    if all_invalid:
        ids = [-1, 256] * 4
    scores = [float("nan"), -0.5, float("inf"), 1.25, -float("inf"), 0.25, 8.0, 0.0]
    # Rotate slots to exercise different FC2 warps and T=4 route permutations.
    rows = [ids[t % 8 :] + ids[: t % 8] for t in range(tokens)]
    vals = [scores[t % 8 :] + scores[: t % 8] for t in range(tokens)]
    got, reason = moe.maybe_run_specialized(
        x,
        torch.tensor(rows, dtype=torch.int32, device=device),
        torch.tensor(vals, dtype=torch.float32, device=device),
        w13,
        w2,
        scales,
        None,
        True,
    )
    assert reason is None
    # Closed-form oracle: both FC1 projections of the one-hot input are 1;
    # each FC2 row sums 512 identical bf16 intermediates. Select routes by
    # their original ID before doing any arithmetic with the score.
    inter = torch.nn.functional.silu(torch.tensor(1.0)).to(torch.bfloat16).item()
    expected = [
        sum(
            512 * inter * (2 if e == 255 else 1) * s
            for e, s in zip(row, val, strict=True)
            if 0 <= e < 256
        )
        for row, val in zip(rows, vals, strict=True)
    ]
    ref = torch.tensor(expected, dtype=x.dtype, device=device)[:, None].expand_as(x)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("entry", ["warmup", "dispatch"])
def test_module_lookup_is_inside_operand_device_scope(weights, monkeypatch, entry):
    w13, w2, scales = weights
    real_device = torch.cuda.device
    entered = []

    class observed_device(real_device):
        def __enter__(self):
            result = super().__enter__()
            entered.append(torch.cuda.current_device())
            return result

        def __exit__(self, *args):
            try:
                return super().__exit__(*args)
            finally:
                entered.pop()

    class LookupReached(Exception):
        pass

    def lookup():
        assert entered and entered[-1] == w13.device.index
        raise LookupReached

    monkeypatch.delenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", raising=False)
    monkeypatch.setattr(torch.cuda, "device", observed_device)
    monkeypatch.setattr(moe, "_kernel_entry", lookup)
    with pytest.raises(LookupReached):
        if entry == "warmup":
            moe.warmup(w13, w2, scales, t_values=[1])
        else:
            x = torch.zeros((1, 2048), device=w13.device, dtype=torch.bfloat16)
            ids = torch.zeros((1, 8), device=w13.device, dtype=torch.int32)
            scores = torch.ones((1, 8), device=w13.device, dtype=torch.float32)
            moe.maybe_run_specialized(x, ids, scores, w13, w2, scales, None, True)
    assert not entered
