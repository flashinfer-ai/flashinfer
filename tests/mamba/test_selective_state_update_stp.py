import numpy as np
import pytest
import torch

import flashinfer

from .selective_state_update_triton import selective_state_update_triton
from .test_utils import create_test_inputs


def clone_preserving_strides(tensor):
    """Clone a tensor while preserving its strides (non-contiguous layout)."""
    result = torch.empty_strided(
        tensor.size(), tensor.stride(), dtype=tensor.dtype, device=tensor.device
    )
    result.copy_(tensor)
    return result


class TestSelectiveStateUpdate:
    """Test class for selective state update kernels."""

    # Test configuration
    ATOL = 1e-3
    RTOL = 1e-2
    NGROUPS = 8
    INPUT_DTYPE = torch.bfloat16
    MATRIX_A_DTYPE = torch.float32

    @pytest.fixture(params=[1, 64])
    def batch(self, request):
        return request.param

    @pytest.fixture(params=[8, 64])
    def nheads(self, request):
        return request.param

    @pytest.fixture(params=[64, 128])
    def dim(self, request):
        return request.param

    @pytest.fixture(params=[64, 128, 256])
    def dstate(self, request):
        return request.param

    @pytest.fixture(params=[torch.float16, torch.bfloat16, torch.float32])
    def state_dtype(self, request):
        return request.param

    @pytest.fixture(params=[torch.float32, torch.bfloat16])
    def weight_dtype(self, request):
        return request.param

    @pytest.fixture(params=[False, True])
    def use_out_tensor(self, request):
        return request.param

    @pytest.fixture
    def inputs(self, batch, nheads, dim, dstate, state_dtype, weight_dtype):
        """Create test inputs for given parameters."""
        return create_test_inputs(
            batch,
            nheads,
            dim,
            dstate,
            self.NGROUPS,
            self.INPUT_DTYPE,
            weight_dtype=weight_dtype,
            matrixA_dtype=self.MATRIX_A_DTYPE,
            state_dtype=state_dtype,
            generate_z=False,
            seed=0,
        )

    @pytest.fixture
    def reference_output(self, inputs):
        """Compute reference output using triton implementation."""
        state_ref = inputs["state_cache"].clone()
        y_ref = selective_state_update_triton(
            state_ref,
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            D=inputs["D"],
            z=inputs.get("z"),
            dt_bias=inputs["dt_bias"],
            dt_softplus=True,
            state_batch_indices=inputs["slot_idx"],
            pad_slot_id=-1,
        )
        return y_ref, state_ref

    def run_kernel(self, inputs, out=None):
        """Run the flashinfer kernel and return output."""
        return flashinfer.mamba.selective_state_update(
            inputs["state_cache"],
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            D=inputs["D"],
            z=inputs.get("z"),
            dt_bias=inputs["dt_bias"],
            dt_softplus=True,
            state_batch_indices=inputs["slot_idx"],
            pad_slot_id=-1,
            out=out,
        )

    def assert_outputs_match(self, y_ref, y_test, msg_prefix=""):
        """Assert outputs match with detailed error reporting."""
        outputs_match = torch.allclose(y_ref, y_test, atol=self.ATOL, rtol=self.RTOL)

        if outputs_match:
            print(
                f"✓ {msg_prefix}Outputs match within tolerance (atol={self.ATOL}, rtol={self.RTOL})"
            )
        else:
            print(
                f"✗ {msg_prefix}Outputs do NOT match within tolerance (atol={self.ATOL}, rtol={self.RTOL})"
            )
            self._print_mismatch_details(y_ref, y_test, "output")

        assert outputs_match

    def assert_states_match(self, state_ref, state_test, slot_idx, msg_prefix=""):
        """Assert states match with detailed error reporting."""
        state_ref_batch = state_ref[slot_idx]
        state_test_batch = state_test[slot_idx]
        states_match = torch.allclose(
            state_ref_batch, state_test_batch, atol=self.ATOL, rtol=self.RTOL
        )

        if states_match:
            print(
                f"✓ {msg_prefix}States match within tolerance (atol={self.ATOL}, rtol={self.RTOL})"
            )
        else:
            print(
                f"✗ {msg_prefix}States do NOT match within tolerance (atol={self.ATOL}, rtol={self.RTOL})"
            )
            self._print_mismatch_details(state_ref_batch, state_test_batch, "state")

        assert states_match

    def _print_mismatch_details(self, ref, test, name):
        """Print detailed mismatch analysis."""
        ref_np = ref.detach().cpu().float().numpy()
        test_np = test.detach().cpu().float().numpy()

        mismatch_mask = ~np.isclose(ref_np, test_np, atol=self.ATOL, rtol=self.RTOL)
        num_mismatches = np.sum(mismatch_mask)
        total_elements = ref_np.size

        print(f"\nDetailed {name} mismatch analysis:")
        print(
            f"Number of mismatched elements: {num_mismatches} / {total_elements} "
            f"({100 * num_mismatches / total_elements:.2f}%)"
        )

        mismatch_indices = np.argwhere(mismatch_mask)
        print(f"First few {name} mismatch locations (up to 10):")
        for idx in mismatch_indices[:10]:
            idx_tuple = tuple(int(i) for i in idx)
            ref_val = ref_np[idx_tuple]
            test_val = test_np[idx_tuple]
            diff = abs(ref_val - test_val)
            rel_diff = diff / (abs(ref_val) + 1e-8)
            print(
                f"  Index {idx_tuple}: ref={ref_val:.6f}, test={test_val:.6f}, "
                f"diff={diff:.6e}, rel_diff={rel_diff:.6e}"
            )

    def test_output_correctness(self, inputs, reference_output, use_out_tensor):
        """Test that kernel output matches reference within tolerance."""
        y_ref, state_ref = reference_output

        # Prepare output tensor if requested
        if use_out_tensor:
            batch = inputs["x"].shape[0]
            nheads = inputs["x"].shape[1]
            dim = inputs["x"].shape[2]
            out = torch.empty(batch, nheads, dim, dtype=self.INPUT_DTYPE, device="cuda")
        else:
            out = None

        y_test = self.run_kernel(inputs, out=out)

        # Verify output tensor identity if provided
        if use_out_tensor:
            assert y_test.data_ptr() == out.data_ptr(), (
                "Returned tensor should be the same object as the provided output tensor"
            )

        self.assert_outputs_match(y_ref, y_test)
        self.assert_states_match(state_ref, inputs["state_cache"], inputs["slot_idx"])


class TestSelectiveStateUpdateWithZ(TestSelectiveStateUpdate):
    """Test selective_state_update with z tensor (gating)."""

    @pytest.fixture(params=[1])
    def batch(self, request):
        return request.param

    @pytest.fixture(params=[8])
    def nheads(self, request):
        return request.param

    @pytest.fixture(params=[64])
    def dim(self, request):
        return request.param

    @pytest.fixture(params=[128])
    def dstate(self, request):
        return request.param

    @pytest.fixture(params=[torch.bfloat16])
    def state_dtype(self, request):
        return request.param

    @pytest.fixture(params=[torch.bfloat16])
    def weight_dtype(self, request):
        return request.param

    @pytest.fixture
    def inputs(self, batch, nheads, dim, dstate, state_dtype, weight_dtype):
        """Create test inputs with z tensor."""
        return create_test_inputs(
            batch,
            nheads,
            dim,
            dstate,
            self.NGROUPS,
            self.INPUT_DTYPE,
            weight_dtype=weight_dtype,
            matrixA_dtype=self.MATRIX_A_DTYPE,
            state_dtype=state_dtype,
            generate_z=True,
            seed=0,
        )


@pytest.mark.xfail(reason="Non-contiguous state cache not yet supported")
class TestSelectiveStateUpdateNonContiguous(TestSelectiveStateUpdate):
    """Test selective_state_update with non-contiguous state cache."""

    @pytest.fixture(params=[64, 128])
    def dstate(self, request):
        return request.param

    @pytest.fixture(params=[torch.bfloat16, torch.float32])
    def state_dtype(self, request):
        return request.param

    @pytest.fixture(params=[torch.float32])
    def weight_dtype(self, request):
        return request.param

    @pytest.fixture
    def inputs(self, batch, nheads, dim, dstate, state_dtype, weight_dtype):
        """Create test inputs with non-contiguous state cache (2x batch stride)."""
        noncontiguous_batch_stride = 2 * nheads * dim * dstate

        return create_test_inputs(
            batch,
            nheads,
            dim,
            dstate,
            self.NGROUPS,
            self.INPUT_DTYPE,
            weight_dtype=weight_dtype,
            matrixA_dtype=self.MATRIX_A_DTYPE,
            state_dtype=state_dtype,
            generate_z=False,
            state_cache_batch_stride=noncontiguous_batch_stride,
            seed=0,
        )

    @pytest.fixture
    def reference_output(self, inputs):
        """Compute reference output, preserving non-contiguous strides."""
        state_ref = clone_preserving_strides(inputs["state_cache"])
        y_ref = selective_state_update_triton(
            state_ref,
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            D=inputs["D"],
            z=inputs.get("z"),
            dt_bias=inputs["dt_bias"],
            dt_softplus=True,
            state_batch_indices=inputs["slot_idx"],
            pad_slot_id=-1,
        )
        return y_ref, state_ref
