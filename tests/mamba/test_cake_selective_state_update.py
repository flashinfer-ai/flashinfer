"""Correctness and source-integrity coverage for the Cake backend."""

from __future__ import annotations

import hashlib

import pytest
import torch

from flashinfer.jit.mamba import cake_selective_state_update as cake
from flashinfer.mamba import cake_selective_state_update, selective_state_update


def test_cake_selective_state_update_sources_match_manifest() -> None:
    source_dir = cake._source_dir()
    assert len(cake._PROGRAMS) == 13
    for name, program in cake._PROGRAMS.items():
        device = source_dir / "cuda" / f"cake_selective_state_update_{name}.cu"
        host = source_dir / "host" / f"cake_selective_state_update_{name}.cc"
        assert hashlib.sha256(device.read_bytes()).hexdigest() == program.device_sha256
        assert hashlib.sha256(host.read_bytes()).hexdigest() == program.host_sha256


_CASES = (
    ("stp_ratio1", 32, 16, 16, 128, 128, 0, torch.bfloat16, "auto", False, None),
    ("stp_ratio8", 64, 64, 8, 128, 128, 0, torch.bfloat16, "auto", False, None),
    ("stp_ratio8_sat", 128, 64, 8, 128, 128, 0, torch.bfloat16, "auto", False, None),
    ("stp_ratio16", 32, 128, 8, 128, 128, 0, torch.bfloat16, "auto", False, None),
    ("stp_fp32", 64, 64, 8, 128, 128, 0, torch.float32, "auto", False, None),
    ("mtp_short1", 64, 64, 8, 128, 128, 1, torch.bfloat16, "auto", False, None),
    ("mtp_short2", 64, 64, 8, 128, 128, 2, torch.bfloat16, "auto", False, None),
    ("mtp_cache", 1, 64, 8, 64, 128, 6, torch.bfloat16, "vertical", True, None),
    ("mtp_horizontal", 32, 64, 8, 64, 128, 6, torch.bfloat16, "horizontal", True, None),
    ("dynamic0", 1, 16, 1, 64, 128, 1, torch.float32, "simple", False, 0),
    ("dynamic1", 1, 16, 1, 64, 128, 4, torch.float32, "simple", False, 1),
    ("dynamic3", 1, 16, 1, 64, 128, 8, torch.float32, "simple", False, 3),
    ("dynamic7", 1, 16, 1, 64, 128, 8, torch.float32, "simple", False, 7),
)


def _make_case(case):
    (
        _name,
        batch_size,
        nheads,
        ngroups,
        dim,
        dstate,
        token_steps,
        state_dtype,
        algorithm,
        cache_intermediate,
        checkpoint_step,
    ) = case
    generator = torch.Generator(device="cuda").manual_seed(0)
    state = (
        torch.randn(
            (batch_size, nheads, dim, dstate),
            generator=generator,
            device="cuda",
        )
        * 0.05
    ).to(state_dtype)
    x_shape = (
        (batch_size, nheads, dim)
        if token_steps == 0
        else (batch_size, token_steps, nheads, dim)
    )
    x = (torch.randn(x_shape, generator=generator, device="cuda") * 0.1).to(
        torch.bfloat16
    )
    dt_base = torch.randn(x_shape[:-1], generator=generator, device="cuda")
    dt = dt_base.as_strided(x_shape, (*dt_base.stride(), 0))
    A_base = -torch.rand((nheads,), generator=generator, device="cuda") - 1.0
    A = A_base.as_strided((nheads, dim, dstate), (1, 0, 0))
    bc_shape = (
        (batch_size, ngroups, dstate)
        if token_steps == 0
        else (batch_size, token_steps, ngroups, dstate)
    )
    B = (torch.randn(bc_shape, generator=generator, device="cuda") * 0.1).to(
        torch.bfloat16
    )
    C = (torch.randn(bc_shape, generator=generator, device="cuda") * 0.1).to(
        torch.bfloat16
    )
    D_base = torch.randn((nheads,), generator=generator, device="cuda")
    D = D_base.as_strided((nheads, dim), (1, 0))
    bias_base = torch.rand((nheads,), generator=generator, device="cuda") - 4.0
    dt_bias = bias_base.as_strided((nheads, dim), (1, 0))
    source = torch.arange(batch_size, dtype=torch.int64, device="cuda")
    destination = None
    if checkpoint_step is not None:
        destination = torch.full(
            (batch_size, token_steps), -1, dtype=torch.int64, device="cuda"
        )
        destination[:, checkpoint_step] = source
    intermediate = None
    intermediate_indices = None
    if cache_intermediate:
        intermediate = torch.empty(
            (batch_size, token_steps, nheads, dim, dstate),
            dtype=state_dtype,
            device="cuda",
        )
        intermediate_indices = source
    return {
        "state": state,
        "x": x,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "dt_bias": dt_bias,
        "state_batch_indices": source,
        "dst_state_batch_indices": destination,
        "cache_steps": token_steps,
        "algorithm": algorithm,
        "dt_softplus": checkpoint_step is not None or cache_intermediate,
        "disable_state_update": cache_intermediate,
        "intermediate_states_buffer": intermediate,
        "intermediate_state_indices": intermediate_indices,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case[0])
def test_cake_selective_state_update_matches_flashinfer(case) -> None:
    inputs = _make_case(case)
    reference = dict(inputs)
    candidate = dict(inputs)
    reference["state"] = inputs["state"].clone()
    candidate["state"] = inputs["state"].clone()
    if inputs["intermediate_states_buffer"] is not None:
        reference["intermediate_states_buffer"] = inputs[
            "intermediate_states_buffer"
        ].clone()
        candidate["intermediate_states_buffer"] = inputs[
            "intermediate_states_buffer"
        ].clone()

    out_reference = selective_state_update(**reference, backend="flashinfer")
    out_candidate = cake_selective_state_update(**candidate)
    torch.testing.assert_close(out_candidate, out_reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        candidate["state"], reference["state"], atol=1e-2, rtol=1e-2
    )
    if candidate["intermediate_states_buffer"] is not None:
        torch.testing.assert_close(
            candidate["intermediate_states_buffer"],
            reference["intermediate_states_buffer"],
            atol=1e-2,
            rtol=1e-2,
        )
