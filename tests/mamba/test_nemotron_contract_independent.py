"""Independent Nemotron-shaped SSU contracts.

These cases deliberately use H != D and group-coded B/C values.  Existing
provider-vs-provider tests can miss an axis swap when H and D are both 64.
"""

import pytest
import torch

from flashinfer.mamba import selective_state_update
from flashinfer.mamba.musa_ssu_native import musa_ssu_one_token_native

from .utils import TEST_DEVICE


@pytest.mark.skipif(TEST_DEVICE != "musa", reason="MUSA contract test")
def test_stp_group_mapping_and_src_dst_against_fp32_oracle():
    torch.manual_seed(1901)
    device = torch.device(TEST_DEVICE)
    batch, heads, dim, dstate, groups, slots = 2, 4, 7, 16, 2, 8
    state = torch.randn(
        slots, heads, dim, dstate, device=device, dtype=torch.bfloat16
    )
    original = state.clone()
    x = torch.randn(batch, heads, dim, device=device, dtype=torch.bfloat16)
    dt_base = torch.randn(batch, heads, device=device, dtype=torch.float32) * 0.2
    dt = dt_base[:, :, None].expand(batch, heads, dim)
    a_base = -torch.rand(heads, device=device, dtype=torch.float32) - 1
    A = a_base[:, None, None].expand(heads, dim, dstate)
    B = torch.empty(batch, groups, dstate, device=device, dtype=torch.bfloat16)
    C = torch.empty_like(B)
    for group in range(groups):
        B[:, group].normal_(mean=group + 1.0, std=0.05)
        C[:, group].normal_(mean=-(group + 1.0), std=0.05)
    D = torch.randn(heads, dim, device=device, dtype=torch.float32)
    bias_base = torch.randn(heads, device=device, dtype=torch.float32) - 2
    dt_bias = bias_base[:, None].expand(heads, dim)
    z = torch.randn(batch, heads, dim, device=device, dtype=torch.bfloat16)
    src = torch.tensor([1, 3], device=device, dtype=torch.int32)
    dst = torch.tensor([5, 6], device=device, dtype=torch.int32)
    out = torch.empty_like(x)

    actual = selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=src,
        dst_state_batch_indices=dst,
        out=out,
        backend="flashinfer",
    )

    expected_state = original.float()
    expected = torch.empty_like(x, dtype=torch.float32)
    ratio = heads // groups
    for b in range(batch):
        for h in range(heads):
            g = h // ratio
            delta = torch.nn.functional.softplus(dt_base[b, h] + bias_base[h])
            running = expected_state[src[b], h] * torch.exp(a_base[h] * delta)
            running = running + delta * x[b, h].float()[:, None] * B[b, g].float()[None, :]
            expected_state[dst[b], h] = running
            y = (C[b, g].float()[None, :] * running).sum(dim=-1)
            y = y + D[h] * x[b, h].float()
            expected[b, h] = y * z[b, h].float() * torch.sigmoid(z[b, h].float())

    torch.testing.assert_close(actual.float(), expected, rtol=0.04, atol=0.04)
    torch.testing.assert_close(
        state[dst].float(), expected_state[dst], rtol=0.04, atol=0.04
    )
    torch.testing.assert_close(state[src], original[src], rtol=0, atol=0)


@pytest.mark.skipif(TEST_DEVICE != "musa", reason="MUSA contract test")
@pytest.mark.skipif(
    __import__("os").environ.get("FLASHINFER_MUSA_SIMPLE_STP_NATIVE") != "1",
    reason="native extension is opt-in",
)
def test_native_nemotron_recurrence_against_independent_fp32_oracle():
    torch.manual_seed(1902)
    device = torch.device(TEST_DEVICE)
    heads, dim, dstate, groups, slots = 64, 64, 128, 8, 8
    state = torch.randn(slots, heads, dim, dstate, device=device, dtype=torch.float16)
    original = state.clone()
    x = torch.randn(1, heads, dim, device=device, dtype=torch.bfloat16)
    dt_head = torch.randn(heads, device=device, dtype=torch.float32) * 0.1
    dt = dt_head[None, :, None].expand(1, heads, dim)
    a_head = -torch.rand(heads, device=device, dtype=torch.float32) - 1
    A = a_head[:, None, None].expand(heads, dim, dstate)
    B = torch.empty(1, groups, dstate, device=device, dtype=torch.bfloat16)
    C = torch.empty_like(B)
    for group in range(groups):
        B[:, group].fill_(group + 1)
        C[:, group].fill_(group + 2)
    D = torch.randn(heads, device=device, dtype=torch.bfloat16)
    src = torch.tensor([1], device=device, dtype=torch.int32)
    dst = torch.tensor([6], device=device, dtype=torch.int32)
    out = torch.empty_like(x)
    musa_ssu_one_token_native(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        src,
        dst,
        None,
        None,
        False,
        -1,
        out,
        None,
        0,
    )

    expected_state = original.float()
    expected = torch.empty((1, heads, dim), device=device, dtype=torch.float32)
    for h in range(heads):
        group = h // (heads // groups)
        running = original[1, h].float() * torch.exp(a_head[h] * dt_head[h])
        running = running + dt_head[h] * x[0, h].float()[:, None] * B[0, group].float()[None, :]
        expected_state[6, h] = running
        expected[0, h] = (C[0, group].float()[None, :] * running).sum(-1) + D[h].float() * x[0, h].float()

    torch.testing.assert_close(out.float(), expected, rtol=0.04, atol=0.04)
    torch.testing.assert_close(state[6].float(), expected_state[6], rtol=0.04, atol=0.04)
    torch.testing.assert_close(state[1], original[1], rtol=0, atol=0)


@pytest.mark.skipif(TEST_DEVICE != "musa", reason="MUSA contract test")
def test_varlen_final_states_against_independent_fp32_oracle():
    """Check packed sequence boundaries and every destination state, not only output."""
    torch.manual_seed(1903)
    device = torch.device(TEST_DEVICE)
    lengths = [1, 3]
    max_seqlen = 3
    tokens, heads, dim, dstate, groups, slots = 4, 4, 7, 16, 2, 32
    state = torch.randn(slots, heads, dim, dstate, device=device, dtype=torch.bfloat16)
    original = state.clone()
    x = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16)
    dt_base = torch.randn(tokens, heads, device=device, dtype=torch.float32) * 0.1
    dt = dt_base[:, :, None].expand(tokens, heads, dim)
    a_base = -torch.rand(heads, device=device, dtype=torch.float32) - 1
    A = a_base[:, None, None].expand(heads, dim, dstate)
    B = torch.randn(tokens, groups, dstate, device=device, dtype=torch.bfloat16)
    C = torch.randn_like(B)
    D = torch.randn(heads, dim, device=device, dtype=torch.float32)
    bias_base = torch.randn(heads, device=device, dtype=torch.float32) - 2
    dt_bias = bias_base[:, None].expand(heads, dim)
    cu = torch.tensor([0, 1, 4], device=device, dtype=torch.int32)
    src = torch.tensor([[2, -1, -1], [7, 8, 9]], device=device, dtype=torch.int32)
    dst = torch.tensor([[12, -1, -1], [17, 18, 19]], device=device, dtype=torch.int32)
    out = torch.empty(tokens, heads, dim, device=device, dtype=torch.bfloat16)

    actual = selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=src,
        dst_state_batch_indices=dst,
        pad_slot_id=-1,
        out=out,
        num_accepted_tokens=None,
        cu_seqlens=cu,
        cache_steps=max_seqlen,
        algorithm="simple",
        backend="flashinfer",
    )

    expected_state = original.float()
    expected = torch.empty(tokens, heads, dim, device=device, dtype=torch.float32)
    ratio = heads // groups
    flat = 0
    for seq, length in enumerate(lengths):
        for t in range(length):
            for h in range(heads):
                group = h // ratio
                delta = torch.nn.functional.softplus(dt_base[flat, h] + bias_base[h])
                running = expected_state[src[seq, t], h] * torch.exp(a_base[h] * delta)
                running = running + delta * x[flat, h].float()[:, None] * B[flat, group].float()[None, :]
                expected_state[dst[seq, t], h] = running
                expected[flat, h] = (C[flat, group].float()[None, :] * running).sum(-1) + D[h] * x[flat, h].float()
            flat += 1

    torch.testing.assert_close(actual.float(), expected, rtol=0.05, atol=0.05)
    for slot in [12, 17, 18, 19]:
        torch.testing.assert_close(state[slot].float(), expected_state[slot], rtol=0.05, atol=0.05)
    for slot in [2, 7, 8, 9]:
        torch.testing.assert_close(state[slot], original[slot], rtol=0, atol=0)
