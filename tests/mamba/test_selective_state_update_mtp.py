"""
Multi-Token Prediction (MTP) tests for selective_state_update.

These tests verify the selective_state_update kernel works correctly with
multi-token inputs (batch, T, nheads, dim) for speculative decoding scenarios.
"""

import numpy as np
import pytest
import torch

import flashinfer

from .triton_reference.selective_state_update import selective_state_update_triton
from .utils import create_test_inputs, clone_preserving_strides


# Base combination: batch=64, nheads=64, dim=64, dstate=128, cache_steps=4,
#                   state_dtype=bf16, weight_dtype=f32, use_out_tensor=True
# Each additional row varies exactly one parameter from the base.
# fmt: off
_BASE_PARAMS = [
    # (batch, nheads, dim, dstate, cache_steps, state_dtype,        weight_dtype,   use_out_tensor)
    (  64,    64,     64,  128,    4,           torch.bfloat16,     torch.float32,  True ),  # base
    (   1,    64,     64,  128,    4,           torch.bfloat16,     torch.float32,  True ),  # batch=1
    (   4,    64,     64,  128,    4,           torch.bfloat16,     torch.float32,  True ),  # batch=4
    (  64,     8,     64,  128,    4,           torch.bfloat16,     torch.float32,  True ),  # nheads=8
    (  64,    64,    128,  128,    4,           torch.bfloat16,     torch.float32,  True ),  # dim=128
    (  64,    64,     64,   64,    4,           torch.bfloat16,     torch.float32,  True ),  # dstate=64
    (  64,    64,     64,  128,    1,           torch.bfloat16,     torch.float32,  True ),  # cache_steps=1
    (  64,    64,     64,  128,    8,           torch.bfloat16,     torch.float32,  True ),  # cache_steps=8
    (  64,    64,     64,  128,    4,           torch.float32,      torch.float32,  True ),  # state_dtype=f32
    (  64,    64,     64,  128,    4,           torch.bfloat16,     torch.float32,  False),  # use_out_tensor=False
]
# fmt: on


class TestSelectiveStateUpdateMTP:
    """Test class for multi-token selective state update kernels."""

    # Test configuration
    ATOL = 1e-3
    RTOL = 1e-2
    NGROUPS = 8
    INPUT_DTYPE = torch.bfloat16
    MATRIX_A_DTYPE = torch.float32

    def make_inputs(
        self, batch, nheads, dim, dstate, cache_steps, state_dtype, weight_dtype
    ):
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
            generate_intermediate_states_buffer=False,
            cache_steps=cache_steps,
            seed=0,
        )

    def make_reference_output(self, inputs):
        """Compute reference output using triton implementation."""
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

    def run_kernel(self, inputs, out=None, disable_state_update=False):
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
            disable_state_update=disable_state_update,
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

    @pytest.mark.parametrize(
        "batch,nheads,dim,dstate,cache_steps,state_dtype,weight_dtype,use_out_tensor",
        _BASE_PARAMS,
    )
    def test_output_correctness(
        self,
        batch,
        nheads,
        dim,
        dstate,
        cache_steps,
        state_dtype,
        weight_dtype,
        use_out_tensor,
    ):
        """Test that kernel output matches reference within tolerance."""
        inputs = self.make_inputs(
            batch, nheads, dim, dstate, cache_steps, state_dtype, weight_dtype
        )
        y_ref, state_ref = self.make_reference_output(inputs)

        # Prepare output tensor if requested
        if use_out_tensor:
            out = torch.empty_like(inputs["x"])
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


class TestSelectiveStateUpdateMTPWithZ(TestSelectiveStateUpdateMTP):
    """Test multi-token selective_state_update with z tensor (gating)."""

    def make_inputs(
        self, batch, nheads, dim, dstate, cache_steps, state_dtype, weight_dtype
    ):
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
            generate_intermediate_states_buffer=False,
            cache_steps=cache_steps,
            seed=0,
        )

    @pytest.mark.parametrize(
        "batch,nheads,dim,dstate,cache_steps,state_dtype,weight_dtype,use_out_tensor",
        [(64, 64, 64, 128, 4, torch.bfloat16, torch.float32, True)],
    )
    def test_output_correctness(
        self,
        batch,
        nheads,
        dim,
        dstate,
        cache_steps,
        state_dtype,
        weight_dtype,
        use_out_tensor,
    ):
        super().test_output_correctness(
            batch,
            nheads,
            dim,
            dstate,
            cache_steps,
            state_dtype,
            weight_dtype,
            use_out_tensor,
        )


class TestSelectiveStateUpdateMTPDisableStateUpdate(TestSelectiveStateUpdateMTP):
    """Test multi-token selective_state_update with disable_state_update=True."""

    @pytest.mark.parametrize(
        "batch,nheads,dim,dstate,cache_steps,state_dtype,weight_dtype,use_out_tensor",
        [(64, 64, 64, 128, 4, torch.bfloat16, torch.float32, True)],
    )
    def test_output_correctness(
        self,
        batch,
        nheads,
        dim,
        dstate,
        cache_steps,
        state_dtype,
        weight_dtype,
        use_out_tensor,
    ):
        """Test that kernel output matches reference but state is not updated."""
        inputs = self.make_inputs(
            batch, nheads, dim, dstate, cache_steps, state_dtype, weight_dtype
        )
        y_ref, _ = self.make_reference_output(inputs)

        # Save the initial state before running the kernel
        state_initial = inputs["state_cache"].clone()

        # Prepare output tensor if requested
        if use_out_tensor:
            out = torch.empty_like(inputs["x"])
        else:
            out = None

        y_test = self.run_kernel(inputs, out=out, disable_state_update=True)

        # Verify output tensor identity if provided
        if use_out_tensor:
            assert y_test.data_ptr() == out.data_ptr(), (
                "Returned tensor should be the same object as the provided output tensor"
            )

        # Check that output is still correct
        self.assert_outputs_match(y_ref, y_test, msg_prefix="[disable_state_update] ")

        # Check that state was NOT updated (should remain the same as initial)
        state_after = inputs["state_cache"]
        state_unchanged = torch.allclose(
            state_initial, state_after, atol=1e-8, rtol=1e-8
        )

        if state_unchanged:
            print("✓ [disable_state_update] State cache was not modified (as expected)")
        else:
            print(
                "✗ [disable_state_update] State cache was modified (should remain unchanged!)"
            )
            # Show where state changed
            state_initial_np = state_initial.detach().cpu().float().numpy()
            state_after_np = state_after.detach().cpu().float().numpy()
            mismatch_mask = ~np.isclose(
                state_initial_np, state_after_np, atol=1e-8, rtol=1e-8
            )
            num_changed = np.sum(mismatch_mask)
            print(
                f"Number of changed state elements: {num_changed} / {state_initial_np.size}"
            )

        assert state_unchanged, (
            "State should not be updated when disable_state_update=True"
        )


class TestSelectiveStateUpdateMTPWithIntermediateStates(TestSelectiveStateUpdateMTP):
    """Test multi-token selective_state_update with intermediate states buffer."""

    def make_inputs(
        self, batch, nheads, dim, dstate, cache_steps, state_dtype, weight_dtype
    ):
        """Create test inputs with intermediate states buffer."""
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
            generate_intermediate_states_buffer=True,
            cache_steps=cache_steps,
            seed=0,
        )

    def make_reference_output(self, inputs):
        """Compute reference output using triton implementation with intermediate states."""
        state_ref = clone_preserving_strides(inputs["state_cache"])
        intermediate_states_ref = inputs["intermediate_states_buffer"].clone()

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
            disable_state_update=True,
            intermediate_states_buffer=intermediate_states_ref,
            cache_steps=inputs["cache_steps"],
            intermediate_state_indices=inputs["intermediate_slot_idx"],
        )
        return y_ref, state_ref, intermediate_states_ref

    def run_kernel_with_intermediate_states(self, inputs, out=None):
        """Run the flashinfer kernel with intermediate states buffer."""
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
            disable_state_update=True,
            intermediate_states_buffer=inputs["intermediate_states_buffer"],
            intermediate_state_indices=inputs["intermediate_slot_idx"],
            cache_steps=inputs["cache_steps"],
        )

    # fmt: off
    _INTERMEDIATE_PARAMS = [
        # (batch, nheads, dim, dstate, cache_steps, state_dtype,    weight_dtype,   use_out_tensor)
        (  64,    64,     64,  128,    4,           torch.bfloat16, torch.float32,  True ),  # base
        (  64,    64,     64,   64,    4,           torch.bfloat16, torch.float32,  True ),  # dstate=64
        (  64,    64,     64,  128,    2,           torch.bfloat16, torch.float32,  True ),  # cache_steps=2
        (  64,    64,     64,  128,    8,           torch.bfloat16, torch.float32,  True ),  # cache_steps=8
        (  64,    64,     64,  128,    4,           torch.bfloat16, torch.float32,  False),  # use_out_tensor=False
    ]
    # fmt: on

    @pytest.mark.parametrize(
        "batch,nheads,dim,dstate,cache_steps,state_dtype,weight_dtype,use_out_tensor",
        _INTERMEDIATE_PARAMS,
    )
    def test_output_correctness(
        self,
        batch,
        nheads,
        dim,
        dstate,
        cache_steps,
        state_dtype,
        weight_dtype,
        use_out_tensor,
    ):
        """Test that kernel output matches and intermediate states are cached correctly."""
        inputs = self.make_inputs(
            batch, nheads, dim, dstate, cache_steps, state_dtype, weight_dtype
        )
        y_ref, _state_ref, intermediate_states_ref = self.make_reference_output(inputs)

        # Prepare output tensor if requested
        if use_out_tensor:
            out = torch.empty_like(inputs["x"])
        else:
            out = None

        y_test = self.run_kernel_with_intermediate_states(inputs, out=out)

        # Verify output tensor identity if provided
        if use_out_tensor:
            assert y_test.data_ptr() == out.data_ptr(), (
                "Returned tensor should be the same object as the provided output tensor"
            )

        # Check output
        self.assert_outputs_match(y_ref, y_test, msg_prefix="[intermediate_states] ")

        # Check intermediate states were cached correctly
        cache_steps = inputs["cache_steps"]
        intermediate_states_test = inputs["intermediate_states_buffer"]

        for t in range(cache_steps):
            cached_state_ref = intermediate_states_ref[:, t, :, :, :]
            cached_state_test = intermediate_states_test[:, t, :, :, :]

            states_match = torch.allclose(
                cached_state_ref, cached_state_test, atol=self.ATOL, rtol=self.RTOL
            )

            max_diff = (cached_state_ref - cached_state_test).abs().max().item()
            if states_match:
                print(f"✓ Intermediate state {t} matches (max_diff={max_diff:.6e})")
            else:
                print(f"✗ Intermediate state {t} mismatch (max_diff={max_diff:.6e})")
                self._print_mismatch_details(
                    cached_state_ref, cached_state_test, f"intermediate_state_{t}"
                )

            assert states_match, f"Intermediate state at step {t} mismatch"


class TestSelectiveStateUpdateMTPNonContiguous(TestSelectiveStateUpdateMTP):
    """Test multi-token selective_state_update with non-contiguous state cache."""

    def make_inputs(
        self, batch, nheads, dim, dstate, cache_steps, state_dtype, weight_dtype
    ):
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
            generate_intermediate_states_buffer=False,
            state_cache_batch_stride=noncontiguous_batch_stride,
            cache_steps=cache_steps,
            seed=0,
        )

    def make_reference_output(self, inputs):
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

    @pytest.mark.parametrize(
        "batch,nheads,dim,dstate,cache_steps,state_dtype,weight_dtype,use_out_tensor",
        [(64, 64, 64, 128, 4, torch.bfloat16, torch.float32, True)],
    )
    def test_output_correctness(
        self,
        batch,
        nheads,
        dim,
        dstate,
        cache_steps,
        state_dtype,
        weight_dtype,
        use_out_tensor,
    ):
        super().test_output_correctness(
            batch,
            nheads,
            dim,
            dstate,
            cache_steps,
            state_dtype,
            weight_dtype,
            use_out_tensor,
        )


class TestSelectiveStateUpdateMTPInt32Indices(TestSelectiveStateUpdateMTP):
    """Test multi-token selective_state_update with int32 state_batch_indices."""

    def run_kernel(self, inputs, out=None, disable_state_update=False):
        """Run the flashinfer kernel with int32 state_batch_indices."""
        # Cast slot_idx to int32
        slot_idx_int32 = inputs["slot_idx"].to(torch.int32)

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
            state_batch_indices=slot_idx_int32,
            pad_slot_id=-1,
            out=out,
            disable_state_update=disable_state_update,
        )

    @pytest.mark.parametrize(
        "batch,nheads,dim,dstate,cache_steps,state_dtype,weight_dtype,use_out_tensor",
        [(64, 64, 64, 128, 4, torch.bfloat16, torch.float32, True)],
    )
    def test_output_correctness(
        self,
        batch,
        nheads,
        dim,
        dstate,
        cache_steps,
        state_dtype,
        weight_dtype,
        use_out_tensor,
    ):
        super().test_output_correctness(
            batch,
            nheads,
            dim,
            dstate,
            cache_steps,
            state_dtype,
            weight_dtype,
            use_out_tensor,
        )


class TestSelectiveStateUpdateMTPVariousNgroups(TestSelectiveStateUpdateMTP):
    """Test multi-token selective_state_update with various ngroups values."""

    # fmt: off
    _NGROUPS_PARAMS = [
        # (batch, nheads, dim, dstate, cache_steps, state_dtype,    weight_dtype,   use_out_tensor, ngroups)
        (  64,    64,     64,  128,    4,           torch.bfloat16, torch.float32,  True,           1),
        (  64,    64,     64,  128,    4,           torch.bfloat16, torch.float32,  True,           2),
        (  64,    64,     64,  128,    4,           torch.bfloat16, torch.float32,  True,           4),
        (  64,    64,     64,  128,    4,           torch.bfloat16, torch.float32,  True,           8),
    ]
    # fmt: on

    @pytest.mark.parametrize(
        "batch,nheads,dim,dstate,cache_steps,state_dtype,weight_dtype,use_out_tensor,ngroups",
        _NGROUPS_PARAMS,
    )
    def test_output_correctness(
        self,
        batch,
        nheads,
        dim,
        dstate,
        cache_steps,
        state_dtype,
        weight_dtype,
        use_out_tensor,
        ngroups,
    ):
        """Test that kernel output matches reference within tolerance."""
        inputs = create_test_inputs(
            batch,
            nheads,
            dim,
            dstate,
            ngroups,
            self.INPUT_DTYPE,
            weight_dtype=weight_dtype,
            matrixA_dtype=self.MATRIX_A_DTYPE,
            state_dtype=state_dtype,
            generate_z=False,
            generate_intermediate_states_buffer=False,
            cache_steps=cache_steps,
            seed=0,
        )
        y_ref, state_ref = self.make_reference_output(inputs)

        if use_out_tensor:
            out = torch.empty_like(inputs["x"])
        else:
            out = None

        y_test = self.run_kernel(inputs, out=out)

        if use_out_tensor:
            assert y_test.data_ptr() == out.data_ptr(), (
                "Returned tensor should be the same object as the provided output tensor"
            )

        self.assert_outputs_match(y_ref, y_test)
        self.assert_states_match(state_ref, inputs["state_cache"], inputs["slot_idx"])


class TestSelectiveStateUpdateMTPLargeBatch(TestSelectiveStateUpdateMTP):
    """Test multi-token selective_state_update with larger batch sizes."""

    # fmt: off
    _LARGE_BATCH_PARAMS = [
        # (batch, nheads, dim, dstate, cache_steps, state_dtype,    weight_dtype,   use_out_tensor)
        (  16,    64,     64,  128,    4,           torch.bfloat16, torch.float32,  True ),  # batch=16
        ( 256,    64,     64,  128,    4,           torch.bfloat16, torch.float32,  True ),  # batch=256
    ]
    # fmt: on

    @pytest.mark.parametrize(
        "batch,nheads,dim,dstate,cache_steps,state_dtype,weight_dtype,use_out_tensor",
        _LARGE_BATCH_PARAMS,
    )
    def test_output_correctness(
        self,
        batch,
        nheads,
        dim,
        dstate,
        cache_steps,
        state_dtype,
        weight_dtype,
        use_out_tensor,
    ):
        super().test_output_correctness(
            batch,
            nheads,
            dim,
            dstate,
            cache_steps,
            state_dtype,
            weight_dtype,
            use_out_tensor,
        )


class TestSelectiveStateUpdateMTPIndicesDtypeMismatch:
    """Test that selective_state_update fails with dtype mismatch between indices."""

    def test_state_batch_idx_and_intermediate_idx_dtype_mismatch_should_fail(self):
        """Test that state_batch_indices and intermediate_state_indices dtype mismatch raises an error."""
        batch = 4
        nheads = 32
        dim = 64
        dstate = 128
        ngroups = 8
        cache_steps = 4

        # Create inputs with intermediate states buffer
        inputs = create_test_inputs(
            batch,
            nheads,
            dim,
            dstate,
            ngroups,
            input_dtype=torch.bfloat16,
            weight_dtype=torch.float32,
            matrixA_dtype=torch.float32,
            state_dtype=torch.bfloat16,
            generate_z=False,
            generate_intermediate_states_buffer=True,
            cache_steps=cache_steps,
            seed=0,
        )

        # Convert state_batch_indices to int64 (default is typically int64)
        inputs["slot_idx"] = inputs["slot_idx"].to(torch.int64)

        # Convert intermediate_state_indices to int32 (different dtype)
        inputs["intermediate_slot_idx"] = inputs["intermediate_slot_idx"].to(
            torch.int32
        )

        # This should fail due to dtype mismatch between indices
        with pytest.raises((RuntimeError, ValueError)):
            flashinfer.mamba.selective_state_update(
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
                disable_state_update=True,
                intermediate_states_buffer=inputs["intermediate_states_buffer"],
                intermediate_state_indices=inputs["intermediate_slot_idx"],
                cache_steps=inputs["cache_steps"],
            )
