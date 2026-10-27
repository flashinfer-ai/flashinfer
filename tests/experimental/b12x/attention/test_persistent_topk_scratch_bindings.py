from __future__ import annotations

import pytest
import torch

import b12x.attention.dsa_indexer.persistent_topk as persistent_topk_impl
from b12x.attention.dsa_indexer.persistent_topk import (
    B12XPersistentTopK2048Binding,
    B12XPersistentTopK2048ScratchCaps,
    plan_persistent_topk2048_scratch,
)


def _plan():
    return plan_persistent_topk2048_scratch(
        B12XPersistentTopK2048ScratchCaps(
            device="cpu",
            max_rows=4,
            max_stride=16,
        )
    )


def test_persistent_topk2048_scratch_plan_exposes_one_opaque_state_spec() -> None:
    plan = _plan()

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "persistent_topk2048.state"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert specs[0].nbytes == plan.scratch_nbytes


def test_persistent_topk2048_scratch_plan_binds_caller_owned_arena() -> None:
    plan = _plan()
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    logits = torch.empty((3, 16), dtype=torch.float32)
    lengths = torch.full((3,), 16, dtype=torch.int32)

    binding = plan.bind(scratch=scratch, logits=logits, lengths=lengths)

    assert isinstance(binding, B12XPersistentTopK2048Binding)
    assert binding.logits is logits
    assert binding.lengths is lengths
    assert not hasattr(binding, "workspace")
    assert binding.scratch.dtype == torch.int32
    assert binding.scratch.data_ptr() == scratch.data_ptr()


def test_persistent_topk2048_binding_run_uses_function_binding_argument(
    monkeypatch,
) -> None:
    plan = _plan()
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = plan.bind(
        scratch=scratch,
        logits=torch.empty((3, 16), dtype=torch.float32),
        lengths=torch.full((3,), 16, dtype=torch.int32),
    )
    calls = {}
    sentinel = object()

    def fake_run(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr(persistent_topk_impl, "run_persistent_topk2048", fake_run)

    assert binding.run() is sentinel
    assert calls["binding"] is binding


def test_persistent_topk2048_binding_owns_runtime_tensors() -> None:
    plan = _plan()
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    logits = torch.empty((3, 16), dtype=torch.float32)
    binding = plan.bind(
        scratch=scratch,
        logits=logits,
        lengths=torch.full((3,), 16, dtype=torch.int32),
    )

    with pytest.raises(ValueError, match="binding owns runtime tensors"):
        persistent_topk_impl.run_persistent_topk2048(logits, binding=binding)


def test_persistent_topk2048_binding_runs_reference_fallback_on_cpu() -> None:
    plan = _plan()
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    logits = torch.tensor(
        [
            [0.0, 3.0, 2.0, 1.0],
            [9.0, 8.0, 7.0, 6.0],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([4, 2], dtype=torch.int32)
    binding = plan.bind(scratch=scratch, logits=logits, lengths=lengths)

    output = persistent_topk_impl.run_persistent_topk2048(binding=binding)

    assert output.shape == (2, 2048)
    assert set(output[0, :4].tolist()) == {0, 1, 2, 3}
    row_1_valid = output[1][output[1] >= 0]
    assert set(row_1_valid.tolist()) == {0, 1}


def test_persistent_topk2048_entrypoint_requires_inputs_or_binding() -> None:
    with pytest.raises(TypeError, match="requires logits/lengths or binding"):
        persistent_topk_impl.run_persistent_topk2048()
