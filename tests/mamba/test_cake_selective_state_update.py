"""Correctness and source-integrity coverage for the Cake backend."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from flashinfer.jit.mamba import cake_selective_state_update as cake
from flashinfer.mamba import cake_selective_state_update, selective_state_update


def _make_sglang_raw_layout_case(batch_size: int):
    meta = "meta"
    return {
        "state": torch.empty(65, 64, 64, 128, dtype=torch.bfloat16, device=meta),
        "x": torch.empty_strided(
            (batch_size, 6, 64, 64),
            (26112, 4352, 64, 1),
            dtype=torch.bfloat16,
            device=meta,
        ),
        "dt": torch.empty_strided(
            (batch_size, 6, 64, 64),
            (51072, 8512, 1, 0),
            dtype=torch.bfloat16,
            device=meta,
        ),
        "A": torch.empty_strided(
            (64, 64, 128), (1, 0, 0), dtype=torch.float32, device=meta
        ),
        "B": torch.empty_strided(
            (batch_size, 6, 1, 128),
            (26112, 4352, 128, 1),
            dtype=torch.bfloat16,
            device=meta,
        ),
        "C": torch.empty_strided(
            (batch_size, 6, 1, 128),
            (26112, 4352, 128, 1),
            dtype=torch.bfloat16,
            device=meta,
        ),
        "D": torch.empty_strided((64, 64), (1, 0), dtype=torch.bfloat16, device=meta),
        "dt_bias": torch.empty_strided(
            (64, 64), (1, 0), dtype=torch.bfloat16, device=meta
        ),
        "output": torch.empty(batch_size, 6, 64, 64, dtype=torch.bfloat16, device=meta),
        "state_batch_indices": torch.empty(batch_size, dtype=torch.int32, device=meta),
        "dst_state_batch_indices": None,
        "intermediate_states_buffer": torch.empty(
            5, 6, 64, 64, 128, dtype=torch.bfloat16, device=meta
        ),
        "intermediate_state_indices": torch.empty(
            batch_size, dtype=torch.int32, device=meta
        ),
    }


@pytest.mark.parametrize("batch_size", range(1, 5))
def test_sglang_raw_mtp_cache_layout_accepts_observed_graph_batches(
    batch_size,
) -> None:
    assert cake._is_sglang_raw_mtp_cache_layout(
        **_make_sglang_raw_layout_case(batch_size)
    )


@pytest.mark.parametrize(
    "field",
    (
        "x",
        "dt",
        "B",
        "C",
        "D",
        "dt_bias",
        "output",
        "state_batch_indices",
        "intermediate_states_buffer",
        "intermediate_state_indices",
    ),
)
def test_sglang_raw_mtp_cache_layout_rejects_abi_drift(field) -> None:
    inputs = _make_sglang_raw_layout_case(2)
    if field == "x":
        inputs[field] = torch.empty_strided(
            (2, 6, 64, 64),
            (26113, 4352, 64, 1),
            dtype=torch.bfloat16,
            device="meta",
        )
    elif field == "dt":
        inputs[field] = torch.empty_strided(
            (2, 6, 64, 64),
            (51072, 8513, 1, 0),
            dtype=torch.bfloat16,
            device="meta",
        )
    elif field in ("B", "C"):
        inputs[field] = torch.empty_strided(
            (2, 6, 1, 128),
            (26112, 4353, 128, 1),
            dtype=torch.bfloat16,
            device="meta",
        )
    elif field in ("D", "dt_bias"):
        inputs[field] = torch.empty_strided(
            (64, 64), (1, 0), dtype=torch.float32, device="meta"
        )
    elif field == "output":
        inputs[field] = torch.empty_strided(
            (2, 6, 64, 64),
            (24577, 4096, 64, 1),
            dtype=torch.bfloat16,
            device="meta",
        )
    elif field in ("state_batch_indices", "intermediate_state_indices"):
        inputs[field] = torch.empty(2, dtype=torch.int64, device="meta")
    else:
        inputs[field] = torch.empty_strided(
            (5, 6, 64, 64, 128),
            (3145729, 524288, 8192, 128, 1),
            dtype=torch.bfloat16,
            device="meta",
        )
    assert not cake._is_sglang_raw_mtp_cache_layout(**inputs)


def test_sglang_raw_route_does_not_compact_tensor_views(monkeypatch) -> None:
    inputs = _make_sglang_raw_layout_case(2)
    calls = []

    def reject_compaction(*_args):
        raise AssertionError("raw SGLang route must not create compact views")

    monkeypatch.setattr(cake, "_compact_broadcasts", reject_compaction)
    monkeypatch.setattr(cake, "_target_arch", lambda _device: "sm_103a")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda _index: nullcontext())
    monkeypatch.setattr(
        cake,
        "_load_program",
        lambda name, arch, device_index: SimpleNamespace(
            run=lambda *args: calls.append((name, arch, device_index, args))
        ),
    )

    assert cake.try_cake_selective_state_update(
        **inputs,
        z=None,
        pad_slot_id=-1,
        disable_state_update=True,
        state_scale=None,
        intermediate_state_scales=None,
        rand_seed=None,
        cache_steps=6,
        cu_seqlens=None,
        num_accepted_tokens=None,
        algorithm="vertical",
        dt_softplus=True,
    )
    assert len(calls) == 1
    name, arch, device_index, args = calls[0]
    assert (name, arch, device_index) == (
        "mtp_cache_bf16_c4_t6_sglang_raw",
        "sm_103a",
        0,
    )
    assert args[2] is inputs["dt"]
    assert args[3] is inputs["A"]
    assert args[6] is inputs["D"]
    assert args[7] is inputs["dt_bias"]


def test_sglang_raw_route_rejects_disabled_softplus(monkeypatch) -> None:
    inputs = _make_sglang_raw_layout_case(2)

    monkeypatch.setattr(cake, "_target_arch", lambda _device: "sm_103a")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda _index: nullcontext())
    monkeypatch.setattr(
        cake,
        "_load_program",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("fixed-softplus T6 program must not be loaded")
        ),
    )

    assert not cake.try_cake_selective_state_update(
        **inputs,
        z=None,
        pad_slot_id=-1,
        disable_state_update=True,
        state_scale=None,
        intermediate_state_scales=None,
        rand_seed=None,
        cache_steps=6,
        cu_seqlens=None,
        num_accepted_tokens=None,
        algorithm="vertical",
        dt_softplus=False,
    )


def test_cake_selective_state_update_sources_match_manifest() -> None:
    source_dir = cake._source_dir()
    assert len(cake._PROGRAMS) == 14
    for name, program in cake._PROGRAMS.items():
        device = source_dir / "cuda" / f"cake_selective_state_update_{name}.cu"
        host = source_dir / "host" / f"cake_selective_state_update_{name}.cc"
        assert hashlib.sha256(device.read_bytes()).hexdigest() == program.device_sha256
        assert hashlib.sha256(host.read_bytes()).hexdigest() == program.host_sha256


@pytest.mark.parametrize(
    ("destinations", "expected"),
    (
        ([[0, -1], [1, -1]], 0),
        ([[-1, 0], [-1, 1]], 1),
        ([[0, -1], [-1, 1]], None),
        ([[0, 2], [1, -1]], None),
        ([[-1, -1], [1, -1]], None),
    ),
)
def test_uniform_checkpoint_step(destinations, expected) -> None:
    destination = torch.tensor(destinations, dtype=torch.int64)
    assert cake._uniform_checkpoint_step(destination, -1) == expected


def test_uniform_checkpoint_step_rejects_empty_batch() -> None:
    destination = torch.empty((0, 2), dtype=torch.int64)
    assert cake._uniform_checkpoint_step(destination, -1) is None


_CASES = (
    ("stp_ratio1", 32, 16, 16, 128, 128, 0, torch.bfloat16, "auto", False, None),
    ("stp_ratio8", 64, 64, 8, 128, 128, 0, torch.bfloat16, "auto", False, None),
    ("stp_ratio8_sat", 128, 64, 8, 128, 128, 0, torch.bfloat16, "auto", False, None),
    ("stp_ratio16", 32, 128, 8, 128, 128, 0, torch.bfloat16, "auto", False, None),
    (
        "stp_persistent",
        257,
        128,
        8,
        128,
        128,
        0,
        torch.bfloat16,
        "auto",
        False,
        None,
    ),
    ("stp_fp32", 64, 64, 8, 128, 128, 0, torch.float32, "auto", False, None),
    ("mtp_short1", 64, 64, 8, 128, 128, 1, torch.bfloat16, "auto", False, None),
    ("mtp_short2", 64, 64, 8, 128, 128, 2, torch.bfloat16, "auto", False, None),
    ("mtp_cache", 1, 64, 8, 64, 128, 6, torch.bfloat16, "vertical", True, None),
    ("mtp_horizontal", 32, 64, 8, 64, 128, 6, torch.bfloat16, "horizontal", True, None),
    ("dynamic0", 1, 16, 1, 64, 128, 1, torch.float32, "simple", False, 0),
    ("dynamic0_b8", 8, 16, 1, 64, 128, 2, torch.float32, "simple", False, 0),
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
def test_cake_selective_state_update_matches_flashinfer(case, monkeypatch) -> None:
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake selective state update requires SM100 or SM103")

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

    cake_hits = []
    original_try_cake = cake.try_cake_selective_state_update

    def strict_try_cake(**kwargs):
        hit = original_try_cake(**kwargs)
        cake_hits.append(hit)
        return hit

    monkeypatch.setattr(cake, "try_cake_selective_state_update", strict_try_cake)

    out_reference = selective_state_update(**reference, backend="flashinfer")
    out_candidate = cake_selective_state_update(**candidate)
    assert cake_hits == [True], "promoted test row fell back instead of running Cake"
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
