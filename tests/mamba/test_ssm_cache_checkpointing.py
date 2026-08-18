"""Tests for SSM cache checkpointing (xAB fast-forward in STP kernel).

Verifies that the STP kernel's fast-forward loop produces correct output
when given K buffered past tokens. The reference is the Triton SSM kernel
called with all K+1 tokens in a single multi-token pass — both process
tokens in fp32 registers without intermediate bf16 round-trips.
"""

import numpy as np
import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability

from .triton_reference.selective_state_update import selective_state_update_triton
from .utils import create_test_inputs


def _get_algorithms():
    major, _ = get_compute_capability(torch.device("cuda"))
    algos = ["simple"]
    if major >= 9:
        algos.extend(["vertical", "horizontal"])
    return algos


NGROUPS = 8
MATRIX_A_DTYPE = torch.float32

# Base: batch=16, nheads=64, dim=64, dstate=128, K=4, state=bf16, input=bf16, weight=f32
# Each row varies one parameter from base.
# fmt: off
_CHECKPOINT_PARAMS = [
    # (batch, nheads, dim, dstate, xab_length, state_dtype,      input_dtype,      weight_dtype)
    (  16,    64,     64,  128,    1,          torch.bfloat16,   torch.bfloat16,   torch.float32),  # K=1
    (  16,    64,     64,  128,    4,          torch.bfloat16,   torch.bfloat16,   torch.float32),  # K=4 (base)
    (  16,    64,     64,  128,    8,          torch.bfloat16,   torch.bfloat16,   torch.float32),  # K=8
    (   1,    64,     64,  128,    4,          torch.bfloat16,   torch.bfloat16,   torch.float32),  # batch=1
    (  16,     8,     64,  128,    4,          torch.bfloat16,   torch.bfloat16,   torch.float32),  # nheads=8
    (  16,    64,    128,  128,    4,          torch.bfloat16,   torch.bfloat16,   torch.float32),  # dim=128
    (  16,    64,     64,   64,    4,          torch.bfloat16,   torch.bfloat16,   torch.float32),  # dstate=64
    (  16,    64,     64,  128,    4,          torch.float32,    torch.bfloat16,   torch.float32),  # state=f32
    (  16,    64,     64,  128,    4,          torch.float16,    torch.bfloat16,   torch.float32),  # state=f16
    (  16,    64,     64,  128,    4,          torch.bfloat16,   torch.float16,    torch.float32),  # input=f16
    (  16,    64,     64,  128,    4,          torch.bfloat16,   torch.bfloat16,   torch.bfloat16), # weight=bf16
]
# fmt: on


def _make_multi_token_inputs(
    batch, nheads, dim, dstate, total_tokens,
    state_dtype=torch.bfloat16, input_dtype=torch.bfloat16,
    weight_dtype=torch.float32, seed=0,
):
    """Create inputs for total_tokens sequential STP calls."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ssm_state_cache_size = max(384, batch * 10)

    state_cache = torch.randn(
        ssm_state_cache_size, nheads, dim, dstate, dtype=state_dtype, device="cuda"
    )

    all_x = torch.randn(batch, total_tokens, nheads, dim, dtype=input_dtype, device="cuda")
    all_dt_base = torch.randn(batch, total_tokens, nheads, dtype=weight_dtype, device="cuda")
    all_B = torch.randn(batch, total_tokens, NGROUPS, dstate, dtype=input_dtype, device="cuda")
    all_C = torch.randn(batch, total_tokens, NGROUPS, dstate, dtype=input_dtype, device="cuda")

    A_base = -torch.rand(nheads, dtype=MATRIX_A_DTYPE, device="cuda") - 1.0
    A = A_base.as_strided((nheads, dim, dstate), (1, 0, 0))

    D_base = torch.randn(nheads, dtype=weight_dtype, device="cuda")
    D = D_base.as_strided((nheads, dim), (1, 0))

    dt_bias_base = torch.rand(nheads, dtype=weight_dtype, device="cuda") - 4.0
    dt_bias = dt_bias_base.as_strided((nheads, dim), (1, 0))

    slot_idx = torch.randperm(ssm_state_cache_size, dtype=torch.int64, device="cuda")[:batch]

    return {
        "state_cache": state_cache,
        "all_x": all_x,
        "all_dt_base": all_dt_base,
        "all_B": all_B,
        "all_C": all_C,
        "A": A,
        "D": D,
        "dt_bias": dt_bias,
        "slot_idx": slot_idx,
    }


def _get_single_token_dt(dt_base_slice, nheads, dim):
    """Reshape a (batch, nheads) dt tensor to (batch, nheads, dim) with broadcast."""
    batch = dt_base_slice.shape[0]
    return dt_base_slice.as_strided(
        (batch, nheads, dim), (dt_base_slice.stride(0), 1, 0)
    )


def _reference_triton_multi_token(inputs, xab_length):
    """Reference using Triton with all K+1 tokens in one multi-token call.

    The Triton kernel processes all T tokens in fp32 registers without
    intermediate bf16 round-trips — matching our checkpointing kernel's
    semantics exactly.
    """
    dim = inputs["all_x"].shape[-1]
    total_tokens = xab_length + 1

    state_ref = inputs["state_cache"].clone()

    x = inputs["all_x"][:, :total_tokens]
    dt_base = inputs["all_dt_base"][:, :total_tokens]
    dt = dt_base.unsqueeze(-1).expand(-1, -1, -1, dim)
    B = inputs["all_B"][:, :total_tokens]
    C = inputs["all_C"][:, :total_tokens]

    out = selective_state_update_triton(
        state_ref,
        x, dt, inputs["A"], B, C,
        D=inputs["D"],
        dt_bias=inputs["dt_bias"],
        dt_softplus=True,
        state_batch_indices=inputs["slot_idx"],
        pad_slot_id=-1,
    )

    return out[:, -1], state_ref


class TestCheckpointingCorrectness:
    """Test that STP with xAB buffer matches fp32 reference simulation."""

    ATOL = 1e-2
    RTOL = 1e-2

    @pytest.mark.parametrize("algorithm", _get_algorithms())
    @pytest.mark.parametrize(
        "batch,nheads,dim,dstate,xab_length,state_dtype,input_dtype,weight_dtype",
        _CHECKPOINT_PARAMS,
    )
    def test_fast_forward_matches_reference(
        self, batch, nheads, dim, dstate, xab_length,
        state_dtype, input_dtype, weight_dtype, algorithm,
    ):
        """STP with xAB buffer must match Triton multi-token reference."""
        total_tokens = xab_length + 1
        inputs = _make_multi_token_inputs(
            batch, nheads, dim, dstate, total_tokens,
            state_dtype=state_dtype, input_dtype=input_dtype,
            weight_dtype=weight_dtype,
        )

        # --- Triton reference: process all K+1 tokens in one call ---
        y_ref, state_ref = _reference_triton_multi_token(inputs, xab_length)

        # --- Kernel under test ---
        state_test = inputs["state_cache"].clone()

        xab_x = inputs["all_x"][:, :xab_length]
        xab_dt = inputs["all_dt_base"][:, :xab_length]
        xab_B = inputs["all_B"][:, :xab_length]

        x_cur = inputs["all_x"][:, -1]
        dt_cur = _get_single_token_dt(inputs["all_dt_base"][:, -1], nheads, dim)
        B_cur = inputs["all_B"][:, -1]
        C_cur = inputs["all_C"][:, -1]

        y_test = flashinfer.mamba.selective_state_update(
            state_test,
            x_cur, dt_cur, inputs["A"], B_cur, C_cur,
            D=inputs["D"],
            dt_bias=inputs["dt_bias"],
            dt_softplus=True,
            state_batch_indices=inputs["slot_idx"],
            pad_slot_id=-1,
            algorithm=algorithm,
            xab_x=xab_x,
            xab_dt=xab_dt,
            xab_B=xab_B,
        )

        # --- Check output ---
        outputs_match = torch.allclose(y_ref.to(y_test.dtype), y_test, atol=self.ATOL, rtol=self.RTOL)
        if not outputs_match:
            diff = (y_ref.float() - y_test.float()).abs()
            print(f"Output max diff: {diff.max().item():.6e}, mean: {diff.mean().item():.6e}")
        assert outputs_match, (
            f"[{algorithm}] Output mismatch with xab_length={xab_length}"
        )

    @pytest.mark.parametrize("algorithm", _get_algorithms())
    def test_xab_length_zero_matches_vanilla(self, algorithm):
        """xab_length=0 (empty buffer) must be identical to vanilla STP."""
        batch, nheads, dim, dstate = 16, 64, 64, 128
        input_dtype, weight_dtype, state_dtype = torch.bfloat16, torch.float32, torch.bfloat16
        inputs = create_test_inputs(
            batch, nheads, dim, dstate, NGROUPS, input_dtype,
            weight_dtype=weight_dtype, matrixA_dtype=MATRIX_A_DTYPE,
            state_dtype=state_dtype, generate_z=False, seed=0,
        )

        # Vanilla (no xAB)
        state_vanilla = inputs["state_cache"].clone()
        y_vanilla = flashinfer.mamba.selective_state_update(
            state_vanilla, inputs["x"], inputs["dt"], inputs["A"],
            inputs["B"], inputs["C"], D=inputs["D"],
            dt_bias=inputs["dt_bias"], dt_softplus=True,
            state_batch_indices=inputs["slot_idx"], pad_slot_id=-1,
            algorithm=algorithm,
        )

        # With empty xAB buffer (K=0)
        state_xab = inputs["state_cache"].clone()
        xab_x = torch.empty(batch, 0, nheads, dim, dtype=input_dtype, device="cuda")
        xab_dt = torch.empty(batch, 0, nheads, dtype=weight_dtype, device="cuda")
        xab_B = torch.empty(batch, 0, NGROUPS, dstate, dtype=input_dtype, device="cuda")

        y_xab = flashinfer.mamba.selective_state_update(
            state_xab, inputs["x"], inputs["dt"], inputs["A"],
            inputs["B"], inputs["C"], D=inputs["D"],
            dt_bias=inputs["dt_bias"], dt_softplus=True,
            state_batch_indices=inputs["slot_idx"], pad_slot_id=-1,
            algorithm=algorithm,
            xab_x=xab_x, xab_dt=xab_dt, xab_B=xab_B,
        )

        assert torch.equal(y_vanilla, y_xab), f"[{algorithm}] xab_length=0 differs from vanilla"
        assert torch.equal(
            state_vanilla[inputs["slot_idx"]],
            state_xab[inputs["slot_idx"]],
        ), f"[{algorithm}] state differs with xab_length=0"

    @pytest.mark.parametrize("algorithm", _get_algorithms())
    def test_disable_state_update_with_checkpointing(self, algorithm):
        """State must not change when disable_state_update=True, even with xAB buffer."""
        batch, nheads, dim, dstate, xab_length = 16, 64, 64, 128, 4
        total_tokens = xab_length + 1
        inputs = _make_multi_token_inputs(batch, nheads, dim, dstate, total_tokens)

        state_before = inputs["state_cache"].clone()

        xab_x = inputs["all_x"][:, :xab_length]
        xab_dt = inputs["all_dt_base"][:, :xab_length]
        xab_B = inputs["all_B"][:, :xab_length]

        x_cur = inputs["all_x"][:, -1]
        dt_cur = _get_single_token_dt(inputs["all_dt_base"][:, -1], nheads, dim)
        B_cur = inputs["all_B"][:, -1]
        C_cur = inputs["all_C"][:, -1]

        flashinfer.mamba.selective_state_update(
            inputs["state_cache"],
            x_cur, dt_cur, inputs["A"], B_cur, C_cur,
            D=inputs["D"],
            dt_bias=inputs["dt_bias"],
            dt_softplus=True,
            state_batch_indices=inputs["slot_idx"],
            pad_slot_id=-1,
            disable_state_update=True,
            algorithm=algorithm,
            xab_x=xab_x, xab_dt=xab_dt, xab_B=xab_B,
        )

        assert torch.equal(state_before, inputs["state_cache"]), (
            f"[{algorithm}] State was modified despite disable_state_update=True"
        )
