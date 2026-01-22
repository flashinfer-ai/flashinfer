import numpy as np
import pytest
import torch

import flashinfer

from .selective_state_update_triton import selective_state_update_triton


def create_test_inputs(
    batch_size,
    nheads,
    dim,
    dstate,
    ngroups,
    input_dtype,
    weight_dtype,
    matrixA_dtype,
    state_dtype,
    z_none=True,
):
    # Set seed for reproducibility
    torch.manual_seed(0)

    device = torch.device("cuda")

    # if we use the cache, then the state indices are taken from a specific slot
    # so the state in the kernel will have batch as the first dimension, but it will
    # only come from a particular slot; the full tensor first dim is larger
    ssm_state_cache_size = max(384, int(2 * batch_size))

    state_cache = torch.randn(
        ssm_state_cache_size, nheads, dim, dstate, dtype=state_dtype, device=device
    )

    x = torch.randn(batch_size, nheads, dim, dtype=input_dtype, device=device)

    dt = torch.randn(batch_size, nheads, dtype=weight_dtype, device=device).as_strided(
        (batch_size, nheads, dim), (nheads, 1, 0)
    )
    # The dtype of A is separate in nemotron nano v3.
    # A only has one value per head (as discussed in mamba 2 block) hence the strides.
    A_base = torch.rand(nheads, dtype=matrixA_dtype, device=device)
    A = A_base.as_strided((nheads, dim, dstate), (1, 0, 0))
    assert A.stride() == (1, 0, 0)

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

    # Create z tensor if z_none is False
    z = (
        None
        if z_none
        else torch.randn(batch_size, nheads, dim, dtype=input_dtype, device=device)
    )

    return {
        "state_cache": state_cache,
        "x": x,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "z": z,
        "dt_bias": dt_bias,
        "slot_idx": slot_idx,
    }


@pytest.mark.parametrize("batch", [1, 64])
@pytest.mark.parametrize("nheads", [8, 64])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dstate", [64, 128, 256])
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_out_tensor", [False, True])
def test_selective_state_update(
    batch,
    nheads,
    dim,
    dstate,
    state_dtype,
    weight_dtype,
    use_out_tensor,
):
    """Test selective_state_update correctness against reference implementation."""
    ngroups = 8
    delta_softplus = True
    input_dtype = torch.bfloat16
    matrixA_dtype = torch.float32

    inputs = create_test_inputs(
        batch,
        nheads,
        dim,
        dstate,
        ngroups,
        input_dtype,
        weight_dtype,
        matrixA_dtype,
        state_dtype=state_dtype,
        z_none=True,
    )

    state = inputs["state_cache"]
    state_ref = state.clone()
    y_ref = selective_state_update_triton(
        state_ref,
        inputs["x"],
        inputs["dt"],
        inputs["A"],
        inputs["B"],
        inputs["C"],
        D=inputs["D"],
        z=inputs["z"],
        dt_bias=inputs["dt_bias"],
        dt_softplus=delta_softplus,
        state_batch_indices=inputs["slot_idx"],
        pad_slot_id=-1,
    )

    # Prepare output tensor if use_out_tensor is True
    if use_out_tensor:
        out = torch.empty(batch, nheads, dim, dtype=input_dtype, device="cuda")
    else:
        out = None

    y_test = flashinfer.mamba.selective_state_update(
        state,
        inputs["x"],
        inputs["dt"],
        inputs["A"],
        inputs["B"],
        inputs["C"],
        D=inputs["D"],
        z=inputs["z"],
        dt_bias=inputs["dt_bias"],
        dt_softplus=delta_softplus,
        state_batch_indices=inputs["slot_idx"],
        pad_slot_id=-1,
        out=out,
    )

    # Verify the returned tensor is the same object as the provided output tensor
    if use_out_tensor:
        assert y_test.data_ptr() == out.data_ptr(), (
            "Returned tensor should be the same object as the provided output tensor"
        )

    atol = 1e-3
    rtol = 1e-2
    outputs_match = torch.allclose(y_ref, y_test, atol=atol, rtol=rtol)

    if outputs_match:
        print(f"✓ Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"✗ Outputs do NOT match within tolerance (atol={atol}, rtol={rtol})")

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
        print("First few mismatch locations (up to 10):")
        for idx in mismatch_indices[:10]:
            idx_tuple = tuple(int(i) for i in idx)
            ref_val = y_ref_np[idx_tuple]
            test_val = y_test_np[idx_tuple]
            diff = abs(ref_val - test_val)
            rel_diff = diff / (abs(ref_val) + 1e-8)
            print(
                f"  Index {idx_tuple}: ref={ref_val:.6f}, test={test_val:.6f}, diff={diff:.6e}, rel_diff={rel_diff:.6e}"
            )

    assert outputs_match

    # Check if states match within tolerance
    states_match = torch.allclose(
        state_ref[inputs["slot_idx"]],
        state[inputs["slot_idx"]],
        atol=atol,
        rtol=rtol,
    )

    if states_match:
        print(f"✓ States match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"✗ States do NOT match within tolerance (atol={atol}, rtol={rtol})")

        # Detailed comparison using numpy testing
        state_ref_np = state_ref[inputs["slot_idx"]].detach().cpu().float().numpy()
        state_test_np = state[inputs["slot_idx"]].detach().cpu().float().numpy()

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
        print("First few state mismatch locations (up to 10):")
        for idx in state_mismatch_indices[:10]:
            idx_tuple = tuple(int(i) for i in idx)
            ref_val = state_ref_np[idx_tuple]
            test_val = state_test_np[idx_tuple]
            diff = abs(ref_val - test_val)
            rel_diff = diff / (abs(ref_val) + 1e-8)
            print(
                f"  Index{idx_tuple}: ref={ref_val:.6f}, test={test_val:.6f}, diff={diff:.6e}, rel_diff={rel_diff:.6e}"
            )

    assert states_match


@pytest.mark.parametrize("use_out_tensor", [False, True])
def test_selective_state_update_with_z(use_out_tensor):
    """Test selective_state_update with z tensor (not None)."""
    batch = 1
    nheads = 8
    dim = 64
    dstate = 128
    ngroups = 8
    delta_softplus = True
    input_dtype = torch.bfloat16
    weight_dtype = torch.bfloat16
    matrixA_dtype = torch.float32
    state_dtype = torch.bfloat16

    inputs = create_test_inputs(
        batch,
        nheads,
        dim,
        dstate,
        ngroups,
        input_dtype,
        weight_dtype,
        matrixA_dtype,
        state_dtype=state_dtype,
        z_none=False,
    )

    state = inputs["state_cache"]
    state_ref = state.clone()
    y_ref = selective_state_update_triton(
        state_ref,
        inputs["x"],
        inputs["dt"],
        inputs["A"],
        inputs["B"],
        inputs["C"],
        D=inputs["D"],
        z=inputs["z"],
        dt_bias=inputs["dt_bias"],
        dt_softplus=delta_softplus,
        state_batch_indices=inputs["slot_idx"],
        pad_slot_id=-1,
    )

    # Prepare output tensor if use_out_tensor is True
    if use_out_tensor:
        out = torch.empty(batch, nheads, dim, dtype=input_dtype, device="cuda")
    else:
        out = None

    y_test = flashinfer.mamba.selective_state_update(
        state,
        inputs["x"],
        inputs["dt"],
        inputs["A"],
        inputs["B"],
        inputs["C"],
        D=inputs["D"],
        z=inputs["z"],
        dt_bias=inputs["dt_bias"],
        dt_softplus=delta_softplus,
        state_batch_indices=inputs["slot_idx"],
        pad_slot_id=-1,
        out=out,
    )

    # Verify the returned tensor is the same object as the provided output tensor
    if use_out_tensor:
        assert y_test.data_ptr() == out.data_ptr(), (
            "Returned tensor should be the same object as the provided output tensor"
        )

    atol = 1e-3
    rtol = 1e-2
    torch.testing.assert_close(y_ref, y_test, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        state_ref[inputs["slot_idx"]],
        state[inputs["slot_idx"]],
        atol=atol,
        rtol=rtol,
    )
