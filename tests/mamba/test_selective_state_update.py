import numpy as np
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from .selective_state_update_triton import selective_state_update_triton

import flashinfer


def selective_state_update_ref(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    """
    Argument:
        state: (batch, dstate, dim) or (batch, nheads, dstate, dim)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dstate, dim) or (nheads, dstate, dim)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dstate, dim = state.shape

    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dstate, dim)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape

    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt
    dA = torch.exp(
        rearrange(dt, "b h d -> b h 1 d") * A
    )  # (batch, nheads, dstate, dim)
    B = repeat(B, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    C = repeat(C, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    dB = rearrange(dt, "b h d -> b h 1 d") * rearrange(
        B.float(), "b h n -> b h n 1"
    )  # (batch, nheads, dstate, dim)
    state_new = state.float() * dA + dB * rearrange(
        x.float(), "b h d -> b h 1 d"
    )  # (batch, nheads, dstate, dim)
    state.copy_(state_new.to(state.dtype))
    out = torch.einsum("bhnd,bhn->bhd", state_new, C.float())
    if D is not None:
        out += x.float() * D
    out = (out if z is None else out * F.silu(z.float())).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out


def create_test_inputs(
    batch_size, nheads, dim, dstate, ngroups, input_dtype, weight_dtype, matrixA_dtype
):
    # Set seed for reproducibility
    torch.manual_seed(0)

    device = torch.device("cuda")

    # if we use the cache, then the state indices are taken from a specific slot
    # so the state in the kernel will have batch as the first dimension, but it will
    # only come from a particular slot; the full tensor first dim is larger
    ssm_state_cache_size = max(384, int(2 * batch_size))

    state_cache = torch.randn(
        ssm_state_cache_size, nheads, dim, dstate, dtype=input_dtype, device=device
    )

    x = torch.randn(batch_size, nheads, dim, dtype=input_dtype, device=device)

    dt = torch.randn(batch_size, nheads, dtype=weight_dtype, device=device).as_strided(
        (batch_size, nheads, dim), (nheads, 1, 0)
    )

    A_base = -torch.rand(nheads, dtype=torch.float32, device=device)
    A = A_base.as_strided((nheads, dim, dstate), (1, 0, 0))

    # A = -torch.rand(nheads, dtype=torch.float32, device=device).as_strided(
    #     (nheads, dim, dstate), (1, 0, 0)
    # )
    # print(f"A dtype: {A.dtype}, shape: {A.shape}, stride {A.stride()}")
    assert(A.stride() == (1, 0, 0))

    # B and C - (batch_size, ngroups, dstate)
    B = torch.randn(batch_size, ngroups, dstate, dtype=input_dtype, device=device)
    C = torch.randn(batch_size, ngroups, dstate, dtype=input_dtype, device=device)

    # D - (nheads, dim) with strides (1, 0) - one value per head
    D = torch.randn(nheads, dtype=weight_dtype, device=device).as_strided(
        (nheads, dim), (1, 0)
    )

    dt_bias = torch.randn(nheads, dtype=weight_dtype, device=device).as_strided(
        (nheads, dim), (1, 0)
    )

    # Slot indices for state batching - (batch_size,)
    slot_idx = torch.randperm(ssm_state_cache_size, dtype=torch.int32, device=device)[
        :batch_size
    ]

    return {
        "state_cache": state_cache,
        "x": x,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "dt_bias": dt_bias,
        "slot_idx": slot_idx,
    }

# @pytest.mark.parametrize("batch", [1, 32, 64])
# @pytest.mark.parametrize("nheads", [8, 64])
# @pytest.mark.parametrize("dim", [64])
# @pytest.mark.parametrize("dstate", [128])
# @pytest.mark.parametrize("ngroups", [8])
# @pytest.mark.parametrize("delta_softplus", [True])
# @pytest.mark.parametrize("input_dtype", [torch.bfloat16])
# @pytest.mark.parametrize("weight_dtype", [torch.bfloat16])
# @pytest.mark.parametrize("matrixA_dtype", [torch.float32])
@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("nheads", [64])
@pytest.mark.parametrize("dim", [64])
@pytest.mark.parametrize("dstate", [128])
@pytest.mark.parametrize("ngroups", [8])
@pytest.mark.parametrize("delta_softplus", [True])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16])
@pytest.mark.parametrize("matrixA_dtype", [torch.float32])
def test_selective_state_update(
    batch,
    nheads,
    dim,
    dstate,
    ngroups,
    delta_softplus,
    input_dtype,
    weight_dtype,
    matrixA_dtype,
):
    """Test selective_state_update correctness against reference implementation."""
    # TODO: Create input tensors based on your kernel requirements
    # Example:
    inputs = create_test_inputs(
        batch,
        nheads,
        dim,
        dstate,
        ngroups,
        input_dtype,
        weight_dtype,
        matrixA_dtype,
    )

    # state = inputs["state_cache"][inputs["slot_idx"]]
    # state_ref = rearrange(state, "... p n -> ... n p").detach().clone()
    state = inputs["state_cache"]
    state_ref = state.clone()
    # A = inputs["A"]

    # A_ref = (rearrange(A, "... p n -> ... n p") if A.ndim == 3 else A).detach().clone()
    # y_ref = selective_state_update_ref(
    #     state_ref,
    #     inputs["x"],
    #     inputs["dt"],
    #     A_ref,
    #     inputs["B"],
    #     inputs["C"],
    #     D=inputs["D"],
    #     z=None,
    #     dt_bias=inputs["dt_bias"],
    #     dt_softplus=delta_softplus,
    # )
    y_ref = selective_state_update_triton(
            state_ref,
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            D=inputs["D"],
            z=None,
            dt_bias=inputs["dt_bias"],
            dt_softplus=delta_softplus,
            state_batch_indices=inputs["slot_idx"],
            pad_slot_id=-1,
        )

    y_test =  flashinfer.mamba.selective_state_update(
        state,
        inputs["x"],
        inputs["dt"],
        inputs["A"],
        inputs["B"],
        inputs["C"],
        D=inputs["D"],
        z=None,
        dt_bias=inputs["dt_bias"],
        dt_softplus=delta_softplus,
        state_batch_indices=inputs["slot_idx"],
        pad_slot_id=-1,
    )

    atol=1e-3
    rtol=1e-2
    outputs_match = torch.allclose(y_ref, y_test, atol=atol, rtol=rtol)

    if outputs_match:
        print(f"✓ Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"✗ Outputs do NOT match within tolerance (atol={atol}, rtol={rtol})")
        all_passed = False

        # Detailed comparison using numpy testing
        y_ref_np = y_ref.detach().cpu().float().numpy()
        y_test_np = y_test.detach().cpu().float().numpy()
        print(f"dtypes: ref {y_ref_np.dtype}, test {y_test_np.dtype}")

        print("\nDetailed mismatch analysis:")
        mismatch_mask = ~np.isclose(y_ref_np, y_test_np, atol=atol, rtol=rtol)
        num_mismatches = np.sum(mismatch_mask)
        total_elements = y_ref_np.size

        print(
            f"Number of mismatched elements: {num_mismatches} / {total_elements} ({100 * num_mismatches / total_elements:.2f}%)"
        )

        mismatch_indices = np.argwhere(mismatch_mask)
        print(f"First few mismatch locations (up to 10):")
        for i, idx in enumerate(mismatch_indices[:10]):
            idx_tuple = tuple(idx)
            ref_val = y_ref_np[idx_tuple]
            test_val = y_test_np[idx_tuple]
            diff = abs(ref_val - test_val)
            rel_diff = diff / (abs(ref_val) + 1e-8)
            print(
                f"  Index {idx_tuple}: ref={ref_val:.6f}, test={test_val:.6f}, diff={diff:.6e}, rel_diff={rel_diff:.6e}"
            )

    assert(outputs_match)

    # Compare states (updated in-place)
    print("\nComparing states with reference...")


    state_test = state[inputs["slot_idx"]]
    state_diff = torch.abs(state_ref - state_test)
    max_state_diff = state_diff.max().item()
    mean_state_diff = state_diff.mean().item()

    print(f"Max absolute state difference: {max_state_diff:.6e}")
    print(f"Mean absolute state difference: {mean_state_diff:.6e}")

    # Check if states match within tolerance
    states_match = torch.allclose( state_ref, state_test, atol=atol, rtol=rtol )

    if states_match:
        print(f"✓ States match within tolerance (atol={atol}, rtol={rtol})")
        return True
    else:
        print(f"✗ States do NOT match within tolerance (atol={atol}, rtol={rtol})")

        # Detailed comparison using numpy testing
        state_ref_np = state_ref.detach().cpu().float().numpy()
        state_test_np = state_test.detach().cpu().float().numpy()

        print("\nDetailed state mismatch analysis:")
        state_mismatch_mask = ~np.isclose(
            state_ref_np, state_test_np, atol=atol, rtol=rtol
        )
        num_state_mismatches = np.sum(state_mismatch_mask)
        total_state_elements = state_ref_np.size

        print(
            f"Number of mismatched state elements: {num_state_mismatches} / {total_state_elements} ({100 * num_state_mismatches / total_state_elements:.2f}%)"
        )

        state_mismatch_indices = np.argwhere(state_mismatch_mask)
        print(f"First few state mismatch locations (up to 10):")
        for i, idx in enumerate(state_mismatch_indices[:10]):
            idx_tuple = tuple(idx)
            ref_val = state_ref_np[idx_tuple]
            test_val = state_test_np[idx_tuple]
            diff = abs(ref_val - test_val)
            rel_diff = diff / (abs(ref_val) + 1e-8)
            print(
                f"  Index {idx_tuple}: ref={ref_val:.6f}, test={test_val:.6f}, diff={diff:.6e}, rel_diff={rel_diff:.6e}"
            )
        return False

    assert(states_match)
