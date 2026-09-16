"""S5000 Simple STP SSU provider matches the generic MUSA provider."""

import os

import pytest
import torch

from flashinfer.mamba.musa_ssu_simple import ssu_one_token_musa_simple
from flashinfer.mamba.musa_ssu_triton import ssu_one_token_musa_triton

pytestmark = pytest.mark.skipif(
    os.environ.get("FLASHINFER_MAMBA_TEST_DEVICE") != "musa",
    reason="requires a MUSA runtime",
)


def test_simple_stp_matches_generic_nemotron_shape():
    torch.manual_seed(17)
    batch, heads, dim, dstate, groups = 1, 64, 64, 128, 8
    state = torch.randn(2, heads, dim, dstate, device="musa", dtype=torch.float16)
    x = torch.randn(batch, heads, dim, device="musa", dtype=torch.bfloat16)
    dt = torch.randn(batch, heads, 1, device="musa", dtype=torch.bfloat16).expand(
        batch, heads, dim
    )
    a = -torch.rand(heads, 1, 1, device="musa", dtype=torch.bfloat16).expand(
        heads, dim, dstate
    )
    b = torch.randn(batch, groups, dstate, device="musa", dtype=torch.bfloat16)
    c = torch.randn_like(b)
    skip = torch.randn(heads, device="musa", dtype=torch.bfloat16)
    src = torch.zeros(batch, device="musa", dtype=torch.int32)
    dst = torch.ones(batch, device="musa", dtype=torch.int32)

    state_simple = state.clone()
    state_generic = state.clone()
    out_simple = torch.empty_like(x)
    out_generic = torch.empty_like(x)
    ssu_one_token_musa_simple(
        state_simple, x, dt, a, b, c, skip, src,
        dst_state_batch_indices=dst, dt_softplus=True, out=out_simple,
    )
    ssu_one_token_musa_triton(
        state_generic, x, dt, a, b, c, skip, src,
        dst_state_batch_indices=dst, dt_softplus=True, out=out_generic,
    )
    torch.testing.assert_close(out_simple, out_generic, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(state_simple, state_generic, atol=2e-2, rtol=2e-2)


def test_simple_stp_stochastic_matches_generic():
    torch.manual_seed(19)
    state = torch.randn((2, 64, 64, 128), device="musa", dtype=torch.float16)
    x = torch.randn((1, 64, 64), device="musa", dtype=torch.bfloat16)
    dt = torch.randn((1, 64, 1), device="musa", dtype=torch.bfloat16).expand(1, 64, 64)
    a = (-torch.rand((64, 1, 1), device="musa", dtype=torch.bfloat16)).expand(64, 64, 128)
    b = torch.randn((1, 8, 128), device="musa", dtype=torch.bfloat16)
    c = torch.randn_like(b)
    d = torch.randn((64,), device="musa", dtype=torch.bfloat16)
    slot = torch.zeros((1,), device="musa", dtype=torch.int32)
    simple_state, generic_state = state.clone(), state.clone()
    simple_out, generic_out = torch.empty_like(x), torch.empty_like(x)
    kwargs = {
        "dt_softplus": True,
        "rand_seed": torch.tensor([1234], device="musa", dtype=torch.int64),
        "philox_rounds": 5,
    }
    ssu_one_token_musa_simple(simple_state, x, dt, a, b, c, d, slot, out=simple_out, **kwargs)
    ssu_one_token_musa_triton(generic_state, x, dt, a, b, c, d, slot, out=generic_out, **kwargs)
    torch.testing.assert_close(simple_out, generic_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(simple_state, generic_state, atol=2e-2, rtol=2e-2)
